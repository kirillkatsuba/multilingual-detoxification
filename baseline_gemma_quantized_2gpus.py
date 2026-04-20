"""
Gemma 3 12B (Quantized) for Multilingual Text Detoxification (Multi-GPU)
=========================================================================

Based on baseline_mt0_2gpus.py pattern.
Uses multiprocessing with ProcessPoolExecutor for parallel inference on 2 GPUs.

Usage:
    python baseline_gemma_quantized_2gpus.py --input_path data/test.tsv
    python baseline_gemma_quantized_2gpus.py --input_path data/test.tsv --use_reranking

Requirements:
    pip install transformers accelerate torch pandas tqdm
    pip install bitsandbytes  # For quantization
    pip install sentence-transformers  # For reranking (optional)

Note: You need to accept Gemma's license on HuggingFace and set HF_TOKEN:
    export HF_TOKEN=your_huggingface_token
    huggingface-cli login
"""

import argparse
import concurrent.futures
import math
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Try to import sentence-transformers for reranking
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

FILE_PATH: str = Path(__file__).resolve()
INPUT_DATA_PATH: str = Path(FILE_PATH.parent, "input_data")
OUTPUT_DATA_PATH: str = Path(FILE_PATH.parent, "output_data")

# Language-specific detoxification instructions
LANG_INSTRUCTIONS = {
    "en": "Remove offensive language and rewrite in neutral English.",
    "ru": "Уберите оскорбительные выражения и перепишите на нейтральном русском.",
    "de": "Entfernen Sie beleidigende Sprache und schreiben Sie auf neutralem Deutsch.",
    "zh": "移除冒犯性语言，用中性中文重写。",
    "es": "Elimina el lenguaje ofensivo y reescribe en español neutral.",
    "fr": "Supprimez le langage offensant et réécrivez en français neutre.",
    "it": "Rimuovi il linguaggio offensivo e riscrivi in italiano neutro.",
    "ja": "攻撃的な言葉を削除し、中立な日本語で書き直してください。",
    "ar": "أزل اللغة المسيئة وأعد كتابتها بالعربية المحايدة.",
    "hi": "आपत्तिजनक भाषा हटाएं और तटस्थ हिंदी में पुनः लिखें।",
    "uk": "Приберіть образливі вирази і перепишіть нейтральною українською.",
    "he": "הסר שפה פוגענית וכתוב מחדש בעברית נייטרלית.",
    "am": "መርዛማ ቋንቋን ያስወግዱ እና በገለልተኛ አማርኛ ዳግም ይጻፉ።",
    "tt": "Рәхимсез сүзләрне алыгыз һәм нейтраль татар телендә яңадан языгыз.",
    "hin": "Offensive language remove karo aur neutral Hinglish mein likho.",
    "default": "Remove offensive language and rewrite in neutral terms.",
}

# Few-shot examples
FEW_SHOT_EXAMPLES = {
    "en": [
        {"toxic": "What the f**k are you doing?", "neutral": "What are you doing?"},
        {"toxic": "This is so d*mn stupid!", "neutral": "This is not very good."},
    ],
    "ru": [
        {"toxic": "Какого хр*на ты делаешь?", "neutral": "Что ты делаешь?"},
    ],
    "de": [
        {"toxic": "Was zur H*lle machst du?", "neutral": "Was machst du?"},
    ],
    "zh": [
        {"toxic": "你在搞什么鬼？", "neutral": "你在做什么？"},
    ],
}

# Gemma prompt template
GEMMA_PROMPT_TEMPLATE = """<start_of_turn>user
You are a text detoxification assistant. Your task is to rewrite toxic or offensive text into neutral, non-offensive language while preserving the original meaning.

Instructions: {instruction}

Examples:
{examples}

Now, detoxify the following text:
Original: {toxic_text}
<end_of_turn>
<start_of_turn>model
Neutral: """


class CandidateReranker:
    """Rerank detoxification candidates based on quality metrics."""

    def __init__(self, device_id: int = 0):
        if SBERT_AVAILABLE:
            self.similarity_model = SentenceTransformer('sentence-transformers/LaBSE')
            if torch.cuda.is_available():
                self.similarity_model = self.similarity_model.to(f"cuda:{device_id}")
        else:
            self.similarity_model = None

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        if self.similarity_model is None:
            return 0.5

        embeddings = self.similarity_model.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)

    def rerank(self, original: str, candidates: List[str]) -> Tuple[str, float]:
        """Rerank candidates and return the best one."""
        scores = []

        for candidate in candidates:
            sim_score = self.compute_similarity(original, candidate)
            len_ratio = min(len(candidate), len(original)) / max(len(candidate), len(original), 1)
            total_score = sim_score * 0.7 + len_ratio * 0.3
            scores.append((candidate, total_score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0], scores[0][1]


class GemmaDetoxifier:
    """
    Gemma 3 12B with quantization for text detoxification.
    Supports specific GPU allocation and 4-bit quantization.
    """

    LANGUAGES = Literal[
        "zh", "es", "ru", "ar", "hi", "uk", "de", "am", 
        "en", "it", "ja", "he", "fr", "tt", "hin",
    ]

    def __init__(
        self, 
        model_name: str = "google/gemma-3-12b-it",
        device_id: int = 0,
        use_4bit: bool = True,
        use_8bit: bool = False,
        max_memory: str = "15GB",
        hf_token: Optional[str] = None,
    ):
        """Initialize the GemmaDetoxifier for a specific GPU."""
        
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device_id}")
        else:
            self.device = torch.device("cpu")

        print(f"[GPU {device_id}] Loading model: {model_name}")
        print(f"[GPU {device_id}] Quantization: 4-bit={use_4bit}, 8-bit={use_8bit}")

        if hf_token:
            os.environ["HF_TOKEN"] = hf_token

        # Quantization config
        quantization_config = None
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif use_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load model
        model_kwargs = {
            "trust_remote_code": True,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = {"": device_id}
            model_kwargs["max_memory"] = {device_id: max_memory}
        else:
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = {"": device_id}

        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        except Exception as e:
            print(f"[GPU {device_id}] Error loading {model_name}: {e}")
            print(f"[GPU {device_id}] Trying fallback: google/gemma-2-9b-it")
            self.model = AutoModelForCausalLM.from_pretrained(
                "google/gemma-2-9b-it", **model_kwargs
            )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side='left',
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize reranker
        self.reranker = CandidateReranker(device_id=device_id) if SBERT_AVAILABLE else None

        self.lang_instructions = LANG_INSTRUCTIONS
        self.examples = FEW_SHOT_EXAMPLES
        
        print(f"[GPU {device_id}] Model loaded successfully!")

    def get_examples_string(self, lang: str, num_examples: int = 2) -> str:
        """Get few-shot examples for a language."""
        examples = self.examples.get(lang, self.examples.get("en", []))
        
        ex_str = ""
        for ex in examples[:num_examples]:
            ex_str += f"- Toxic: {ex['toxic']}\n  Neutral: {ex['neutral']}\n"
        
        return ex_str if ex_str else "No examples available."

    def build_prompt(self, toxic_text: str, lang: str = "en", num_shots: int = 2) -> str:
        """Build the prompt for detoxification."""
        instruction = self.lang_instructions.get(lang, self.lang_instructions["default"])
        examples = self.get_examples_string(lang, num_shots)
        
        prompt = GEMMA_PROMPT_TEMPLATE.format(
            instruction=instruction,
            examples=examples,
            toxic_text=toxic_text,
        )
        
        return prompt

    def detoxify_single(
        self, 
        toxic_text: str, 
        lang: str = "en",
        num_shots: int = 2,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Detoxify a single text."""
        prompt = self.build_prompt(toxic_text, lang, num_shots)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the neutral text
        if "Neutral:" in generated:
            neutral = generated.split("Neutral:")[-1].strip()
        else:
            neutral = generated[len(prompt):].strip()

        # Clean up
        for stop in ["\n\n", "<end_of_turn>", "Original:"]:
            if stop in neutral:
                neutral = neutral.split(stop)[0].strip()

        return neutral

    def detoxify_with_reranking(
        self, 
        toxic_text: str, 
        lang: str = "en",
        num_candidates: int = 3,
        num_shots: int = 2,
    ) -> str:
        """Detoxify with reranking multiple candidates."""
        prompt = self.build_prompt(toxic_text, lang, num_shots)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=num_candidates,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode all candidates
        candidates = []
        for output in outputs:
            generated = self.tokenizer.decode(output, skip_special_tokens=True)
            
            if "Neutral:" in generated:
                neutral = generated.split("Neutral:")[-1].strip()
            else:
                neutral = generated[len(prompt):].strip()

            for stop in ["\n\n", "<end_of_turn>", "Original:"]:
                if stop in neutral:
                    neutral = neutral.split(stop)[0].strip()
            
            candidates.append(neutral)

        # Rerank if available
        if self.reranker:
            best, _ = self.reranker.rerank(toxic_text, candidates)
            return best
        else:
            return candidates[0]  # Return first candidate

    def detoxify_batch(
        self, 
        texts: List[str], 
        langs: List[str], 
        batch_size: int = 2, 
        device_id: int = 0,
        use_reranking: bool = False,
        num_candidates: int = 3,
        num_shots: int = 2,
        temperature: float = 0.7,
    ) -> List[str]:
        """Detoxify a batch of texts with their corresponding languages."""
        
        results = []
        
        for i in tqdm(
            range(0, len(texts), batch_size), 
            desc=f"GPU {device_id} Processing",
            position=device_id,
            leave=True
        ):
            batch_texts = texts[i : i + batch_size]
            batch_langs = langs[i : i + batch_size]

            for text, lang in zip(batch_texts, batch_langs):
                if use_reranking:
                    neutral = self.detoxify_with_reranking(
                        toxic_text=text,
                        lang=lang,
                        num_candidates=num_candidates,
                        num_shots=num_shots,
                    )
                else:
                    neutral = self.detoxify_single(
                        toxic_text=text,
                        lang=lang,
                        num_shots=num_shots,
                        temperature=temperature,
                    )
                results.append(neutral)

        return results


# Worker function MUST be at the top level to work with Python multiprocessing
def process_chunk(
    df_chunk: pd.DataFrame, 
    device_id: int, 
    batch_size: int,
    use_4bit: bool = True,
    use_8bit: bool = False,
    use_reranking: bool = False,
    num_candidates: int = 3,
    num_shots: int = 2,
    temperature: float = 0.7,
    hf_token: Optional[str] = None,
) -> pd.DataFrame:
    """Worker function to process a chunk of data on a specific GPU."""
    
    detoxifier = GemmaDetoxifier(
        device_id=device_id,
        use_4bit=use_4bit,
        use_8bit=use_8bit,
        hf_token=hf_token,
    )
    
    neutral_sentences = detoxifier.detoxify_batch(
        texts=df_chunk["toxic_sentence"].tolist(),
        langs=df_chunk["lang"].tolist(),
        batch_size=batch_size,
        device_id=device_id,
        use_reranking=use_reranking,
        num_candidates=num_candidates,
        num_shots=num_shots,
        temperature=temperature,
    )
    
    df_chunk = df_chunk.copy()
    df_chunk["neutral_sentence"] = neutral_sentences
    return df_chunk


def process_file(
    input_path: str, 
    output_path: str, 
    batch_size: int = 2,
    use_4bit: bool = True,
    use_8bit: bool = False,
    use_reranking: bool = False,
    num_candidates: int = 3,
    num_shots: int = 2,
    temperature: float = 0.7,
    hf_token: Optional[str] = None,
):
    """
    Process an input TSV file, split across multiple GPUs, and save output.
    """
    df = pd.read_csv(input_path, sep="\t")
    
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(f"Total rows to process: {len(df)}")
    print(f"Found {num_gpus} device(s) available for processing.")

    if num_gpus > 1:
        chunk_size = math.ceil(len(df) / num_gpus)
        chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
        
        print(f"Split into {len(chunks)} chunks")
        
        results = []
        ctx = mp.get_context('spawn') 
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_gpus, mp_context=ctx) as executor:
            futures = [
                executor.submit(
                    process_chunk, chunk, gpu_id, batch_size,
                    use_4bit, use_8bit, use_reranking, num_candidates,
                    num_shots, temperature, hf_token
                ) 
                for gpu_id, chunk in enumerate(chunks)
            ]
            
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                
        final_df = pd.concat(results).sort_index()
    else:
        final_df = process_chunk(
            df, device_id=0, batch_size=batch_size,
            use_4bit=use_4bit, use_8bit=use_8bit,
            use_reranking=use_reranking, num_candidates=num_candidates,
            num_shots=num_shots, temperature=temperature, hf_token=hf_token
        )

    final_df.to_csv(output_path, sep="\t", index=False)


def main():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Gemma 12B Quantized Detoxification (Multi-GPU)")
    
    parser.add_argument(
        "--input_path",
        type=str,
        default=str(Path(INPUT_DATA_PATH, "dev_inputs.tsv")),
        help="Path to input TSV file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(Path(OUTPUT_DATA_PATH, "gemma_quantized_output.tsv")),
        help="Path to output TSV file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size per GPU for processing (default: 2)",
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        default=True,
        help="Use 4-bit quantization (default: True)",
    )
    parser.add_argument(
        "--use_8bit",
        action="store_true",
        help="Use 8-bit quantization instead of 4-bit",
    )
    parser.add_argument(
        "--use_reranking",
        action="store_true",
        help="Use reranking with multiple candidates",
    )
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=3,
        help="Number of candidates for reranking (default: 3)",
    )
    parser.add_argument(
        "--num_shots",
        type=int,
        default=2,
        help="Number of few-shot examples (default: 2)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature (default: 0.7)",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token for Gemma access",
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Gemma 12B Quantized Detoxification (Multi-GPU)")
    print("=" * 60)
    process_file(
        args.input_path, args.output_path, args.batch_size,
        args.use_4bit and not args.use_8bit, args.use_8bit,
        args.use_reranking, args.num_candidates, args.num_shots,
        args.temperature, args.hf_token
    )
    print(f"\nProcessing complete. Results saved to {output_path}")


if __name__ == "__main__":
    main()
