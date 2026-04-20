#!/usr/bin/env python3
"""
Gemma 3 12B (Quantized) for Multilingual Text Detoxification
=============================================================

This script uses Google's Gemma 3 12B model with 4-bit quantization for text detoxification.
Optimized for 32GB GPU using bitsandbytes quantization.

Features:
- 4-bit quantization for memory efficiency (~24-28GB VRAM)
- Multi-GPU support with device_map="auto"
- Few-shot learning with Chain-of-Thought prompting
- Language-specific prompts and examples
- Reranking with multiple candidates

Usage:
    # Basic inference with 4-bit quantization
    python baseline_gemma_quantized.py --input_path data/test.tsv

    # With reranking (generates multiple candidates)
    python baseline_gemma_quantized.py --use_reranking --num_candidates 3

    # With 8-bit quantization (slightly more accurate, ~28GB VRAM)
    python baseline_gemma_quantized.py --use_8bit

Requirements:
    pip install transformers accelerate bitsandbytes sentencepiece
    pip install sentence-transformers  # For reranking

Note: You need to accept Gemma's license on HuggingFace and set HF_TOKEN:
    export HF_TOKEN=your_huggingface_token
    huggingface-cli login
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Try to import sentence-transformers for reranking
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

# =============================================================================
# Configuration
# =============================================================================

FILE_PATH = Path(__file__).resolve()
INPUT_DATA_PATH = Path(FILE_PATH.parent, "input_data")
OUTPUT_DATA_PATH = Path(FILE_PATH.parent, "output_data")

# Language codes
LANGUAGES = Literal[
    "zh", "es", "ru", "ar", "hi", "uk", "de", "am",
    "en", "it", "ja", "he", "fr", "tt", "hin",
]

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
        {"toxic": "He's such an *sshole!", "neutral": "He is being unpleasant."},
    ],
    "ru": [
        {"toxic": "Какого хр*на ты делаешь?", "neutral": "Что ты делаешь?"},
        {"toxic": "Это п*здец какой-то!", "neutral": "Это очень плохо."},
    ],
    "de": [
        {"toxic": "Was zur H*lle machst du?", "neutral": "Was machst du?"},
        {"toxic": "Das ist totaler Schw*achsinn!", "neutral": "Das ergibt keinen Sinn."},
    ],
    "zh": [
        {"toxic": "你在搞什么鬼？", "neutral": "你在做什么？"},
        {"toxic": "这真是他妈的糟糕！", "neutral": "这真的很糟糕。"},
    ],
}

# Prompt template
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

# Chain-of-Thought prompt template
GEMMA_COT_TEMPLATE = """<start_of_turn>user
You are a text detoxification assistant. Rewrite toxic text into neutral, non-offensive language.

Instructions: {instruction}

{examples}

Detoxify this text step by step:
Original: {toxic_text}

Step 1: Identify toxic words.
Step 2: Find neutral alternatives.
Step 3: Rewrite.
<end_of_turn>
<start_of_turn>model
Let me think through this:

1. Toxic elements: [identifying...]

2. Neutral alternatives: [finding...]

3. Neutral version: """


# =============================================================================
# Reranker for Multiple Candidates
# =============================================================================

class CandidateReranker:
    """Rerank detoxification candidates based on quality metrics."""

    def __init__(self):
        if SBERT_AVAILABLE:
            print("Loading sentence transformer for reranking...")
            self.similarity_model = SentenceTransformer('sentence-transformers/LaBSE')
        else:
            self.similarity_model = None
            print("SentenceTransformer not available. Reranking will use basic scoring.")

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        if self.similarity_model is None:
            return 0.5  # Default score

        embeddings = self.similarity_model.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)

    def compute_length_score(self, text: str, reference: str) -> float:
        """Score based on length similarity."""
        len_ratio = min(len(text), len(reference)) / max(len(text), len(reference), 1)
        return len_ratio

    def rerank(
        self,
        original: str,
        candidates: List[str],
        reference: Optional[str] = None,
    ) -> Tuple[str, float]:
        """Rerank candidates and return the best one."""
        scores = []

        for candidate in candidates:
            # Similarity to original (content preservation)
            sim_score = self.compute_similarity(original, candidate)

            # Length score
            ref = reference or original
            len_score = self.compute_length_score(candidate, ref)

            # Combined score (higher is better)
            # Prefer candidates that preserve meaning (high sim) and are reasonable length
            total_score = sim_score * 0.7 + len_score * 0.3

            scores.append((candidate, total_score, sim_score, len_score))

        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[0][0], scores[0][1]


# =============================================================================
# Gemma Detoxifier
# =============================================================================

class GemmaDetoxifier:
    """Gemma 3 12B with quantization for text detoxification."""

    def __init__(
        self,
        model_name: str = "google/gemma-3-12b-it",  # or "google/gemma-2-9b-it"
        device_id: int = 0,
        use_4bit: bool = True,
        use_8bit: bool = False,
        use_flash_attention: bool = True,
        max_memory: str = "28GB",
        hf_token: Optional[str] = None,
    ):
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit

        print(f"Loading model: {model_name}")
        print(f"Quantization: 4-bit={use_4bit}, 8-bit={use_8bit}")

        # Set HF token if provided
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
            print("Using 4-bit quantization (~24GB VRAM)")
        elif use_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            print("Using 8-bit quantization (~28GB VRAM)")

        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
        }

        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["max_memory"] = {0: max_memory}
        else:
            model_kwargs["torch_dtype"] = torch.float16

        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            print("Trying alternative model: google/gemma-2-9b-it")
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
        self.reranker = CandidateReranker()

        # Language instructions
        self.lang_instructions = LANG_INSTRUCTIONS

        print("Model loaded successfully!")

    def get_examples_string(self, lang: str, num_examples: int = 2) -> str:
        """Get few-shot examples for a language."""
        examples = FEW_SHOT_EXAMPLES.get(lang, FEW_SHOT_EXAMPLES.get("en", []))

        ex_str = ""
        for i, ex in enumerate(examples[:num_examples]):
            ex_str += f"- Toxic: {ex['toxic']}\n  Neutral: {ex['neutral']}\n"

        return ex_str if ex_str else "No examples available."

    def build_prompt(
        self,
        toxic_text: str,
        lang: str = "en",
        use_cot: bool = True,
        num_shots: int = 2,
    ) -> str:
        """Build the prompt for detoxification."""
        instruction = self.lang_instructions.get(lang, self.lang_instructions["default"])
        examples = self.get_examples_string(lang, num_shots)

        if use_cot:
            template = GEMMA_COT_TEMPLATE
        else:
            template = GEMMA_PROMPT_TEMPLATE

        prompt = template.format(
            instruction=instruction,
            examples=examples,
            toxic_text=toxic_text,
        )

        return prompt

    def detoxify_single(
        self,
        toxic_text: str,
        lang: str = "en",
        use_cot: bool = True,
        num_shots: int = 2,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> str:
        """Detoxify a single text."""
        prompt = self.build_prompt(toxic_text, lang, use_cot, num_shots)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the neutral text
        if "Neutral version:" in generated:
            neutral = generated.split("Neutral version:")[-1].strip()
        elif "Neutral:" in generated:
            neutral = generated.split("Neutral:")[-1].strip()
        else:
            # Try to extract after the prompt
            neutral = generated[len(prompt):].strip()

        # Clean up
        for stop in ["\n\n", "<end_of_turn>", "Original:", "Step 1:"]:
            if stop in neutral:
                neutral = neutral.split(stop)[0].strip()

        return neutral

    def detoxify_with_reranking(
        self,
        toxic_text: str,
        lang: str = "en",
        num_candidates: int = 3,
        use_cot: bool = True,
        num_shots: int = 2,
        temperature: float = 0.8,  # Higher temperature for diverse candidates
    ) -> Tuple[str, float]:
        """Detoxify with reranking multiple candidates."""

        prompt = self.build_prompt(toxic_text, lang, use_cot, num_shots)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=temperature,
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

            # Extract neutral text
            if "Neutral version:" in generated:
                neutral = generated.split("Neutral version:")[-1].strip()
            elif "Neutral:" in generated:
                neutral = generated.split("Neutral:")[-1].strip()
            else:
                neutral = generated[len(prompt):].strip()

            # Clean up
            for stop in ["\n\n", "<end_of_turn>", "Original:", "Step 1:"]:
                if stop in neutral:
                    neutral = neutral.split(stop)[0].strip()

            candidates.append(neutral)

        # Rerank
        best_candidate, score = self.reranker.rerank(toxic_text, candidates)

        return best_candidate, score

    def detoxify_batch(
        self,
        texts: List[str],
        langs: List[str],
        batch_size: int = 2,
        use_cot: bool = True,
        num_shots: int = 2,
        use_reranking: bool = False,
        num_candidates: int = 3,
        temperature: float = 0.7,
    ) -> List[str]:
        """Detoxify a batch of texts."""
        results = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Processing"):
            batch_texts = texts[i:i + batch_size]
            batch_langs = langs[i:i + batch_size]

            for text, lang in zip(batch_texts, batch_langs):
                if use_reranking:
                    neutral, _ = self.detoxify_with_reranking(
                        toxic_text=text,
                        lang=lang,
                        num_candidates=num_candidates,
                        use_cot=use_cot,
                        num_shots=num_shots,
                        temperature=0.8,
                    )
                else:
                    neutral = self.detoxify_single(
                        toxic_text=text,
                        lang=lang,
                        use_cot=use_cot,
                        num_shots=num_shots,
                        temperature=temperature,
                    )

                results.append(neutral)

        return results


# =============================================================================
# Main Functions
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Gemma 12B (Quantized) Text Detoxification"
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-3-12b-it",
        help="Model name or path (try google/gemma-2-9b-it as fallback)",
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="GPU device ID",
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
        "--max_memory",
        type=str,
        default="28GB",
        help="Max GPU memory allocation",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token for Gemma access",
    )

    # Prompting arguments
    parser.add_argument(
        "--use_cot",
        action="store_true",
        default=True,
        help="Use Chain-of-Thought prompting",
    )
    parser.add_argument(
        "--num_shots",
        type=int,
        default=2,
        help="Number of few-shot examples",
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
        help="Number of candidates for reranking",
    )

    # Generation arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature",
    )

    # Input/Output arguments
    parser.add_argument(
        "--input_path",
        type=str,
        default=str(Path(INPUT_DATA_PATH, "dev_inputs.tsv")),
        help="Input TSV file path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(Path(OUTPUT_DATA_PATH, "gemma_quantized_output.tsv")),
        help="Output TSV file path",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Gemma 12B (Quantized) Text Detoxification")
    print("=" * 60)

    # Create output directory
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    # Initialize model
    detoxifier = GemmaDetoxifier(
        model_name=args.model_name,
        device_id=args.device_id,
        use_4bit=args.use_4bit and not args.use_8bit,
        use_8bit=args.use_8bit,
        max_memory=args.max_memory,
        hf_token=args.hf_token,
    )

    # Load input data
    df = pd.read_csv(args.input_path, sep='\t')
    print(f"Loaded {len(df)} samples from {args.input_path}")

    # Get languages
    langs = df['lang'].tolist() if 'lang' in df.columns else ['en'] * len(df)

    # Generate
    neutral_sentences = detoxifier.detoxify_batch(
        texts=df['toxic_sentence'].tolist(),
        langs=langs,
        batch_size=args.batch_size,
        use_cot=args.use_cot,
        num_shots=args.num_shots,
        use_reranking=args.use_reranking,
        num_candidates=args.num_candidates,
        temperature=args.temperature,
    )

    # Save results
    df['neutral_sentence'] = neutral_sentences
    df.to_csv(args.output_path, sep='\t', index=False)
    print(f"\nResults saved to {args.output_path}")

    # Print sample results
    print("\nSample results:")
    for i in range(min(3, len(df))):
        print(f"\nOriginal ({df['lang'].iloc[i] if 'lang' in df.columns else 'en'}): {df['toxic_sentence'].iloc[i]}")
        print(f"Detoxified: {df['neutral_sentence'].iloc[i]}")


if __name__ == "__main__":
    main()
