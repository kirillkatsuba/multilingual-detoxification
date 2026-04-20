"""
Qwen2.5-7B Few-Shot Learning for Multilingual Text Detoxification (Multi-GPU)
==============================================================================

Based on baseline_mt0_2gpus.py pattern.
Uses multiprocessing with ProcessPoolExecutor for parallel inference on 2 GPUs.

Usage:
    python baseline_qwen_fewshot_2gpus.py --input_path data/test.tsv
    python baseline_qwen_fewshot_2gpus.py --input_path data/test.tsv --use_4bit

Requirements:
    pip install transformers accelerate torch pandas tqdm
    pip install bitsandbytes  # For 4-bit quantization
"""

import argparse
import concurrent.futures
import math
from pathlib import Path
from typing import Dict, List, Literal, Optional

import pandas as pd
import torch
import torch.multiprocessing as mp
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

FILE_PATH: str = Path(__file__).resolve()
INPUT_DATA_PATH: str = Path(FILE_PATH.parent, "input_data")
OUTPUT_DATA_PATH: str = Path(FILE_PATH.parent, "output_data")

# Few-shot examples per language
FEW_SHOT_EXAMPLES = {
    "en": [
        {"toxic": "What the f**k is this about?", "neutral": "What is this about?"},
        {"toxic": "He's such a d*mn idiot!", "neutral": "He's not very smart."},
        {"toxic": "This is total cr*p!", "neutral": "This is not good."},
    ],
    "ru": [
        {"toxic": "Какого хр*на тут происходит?", "neutral": "Что тут происходит?"},
        {"toxic": "Он такой ид*от!", "neutral": "Он не очень умный."},
    ],
    "de": [
        {"toxic": "Was zur H*lle ist das?", "neutral": "Was ist das?"},
        {"toxic": "Das ist totaler Sch*ss!", "neutral": "Das ist nicht gut."},
    ],
    "zh": [
        {"toxic": "这到底是什么鬼东西？", "neutral": "这到底是什么东西？"},
        {"toxic": "真他妈的烦人！", "neutral": "真烦人！"},
    ],
    "es": [
        {"toxic": "¿Qué m*erda es esto?", "neutral": "¿Qué es esto?"},
    ],
    "fr": [
        {"toxic": "C'est quoi cette m*erde?", "neutral": "C'est quoi ça?"},
    ],
    "default": [
        {"toxic": "This is so d*mn stupid!", "neutral": "This is not very good."},
    ],
}

# Prompt template with few-shot examples
PROMPT_TEMPLATE = """You are a text detoxification assistant. Rewrite the following toxic text in a neutral, non-offensive way while preserving the meaning.

Examples:
{examples}

Toxic text ({language}): {toxic_text}

Neutral text:"""


class QwenDetoxifier:
    """
    Qwen2.5-7B-Instruct for text detoxification with few-shot learning.
    Supports specific GPU allocation and optional quantization.
    """

    LANGUAGES = Literal[
        "zh", "es", "ru", "ar", "hi", "uk", "de", "am", 
        "en", "it", "ja", "he", "fr", "tt", "hin",
    ]

    def __init__(
        self, 
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device_id: int = 0,
        use_4bit: bool = False,
        use_8bit: bool = False,
        max_memory: str = "15GB",
    ):
        """Initialize the QwenDetoxifier for a specific GPU."""
        
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device_id}")
        else:
            self.device = torch.device("cpu")

        print(f"[GPU {device_id}] Loading model: {model_name}")
        print(f"[GPU {device_id}] Quantization: 4-bit={use_4bit}, 8-bit={use_8bit}")

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
            "torch_dtype": torch.float16 if not (use_4bit or use_8bit) else None,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = {"": device_id}
            model_kwargs["max_memory"] = {device_id: max_memory}
        else:
            model_kwargs["device_map"] = {"": device_id}

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side='left',
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Language name mapping
        self.lang_names = {
            "en": "English", "ru": "Russian", "de": "German", "zh": "Chinese",
            "es": "Spanish", "fr": "French", "it": "Italian", "ja": "Japanese",
            "ar": "Arabic", "hi": "Hindi", "uk": "Ukrainian", "he": "Hebrew",
            "am": "Amharic", "tt": "Tatar", "hin": "Hinglish",
        }

        self.examples = FEW_SHOT_EXAMPLES
        
        print(f"[GPU {device_id}] Model loaded successfully!")

    def build_prompt(self, toxic_text: str, lang: str = "en", num_shots: int = 3) -> str:
        """Build the prompt for detoxification."""
        
        language = self.lang_names.get(lang, lang)
        
        # Get examples for the language
        examples = self.examples.get(lang, self.examples.get("en", self.examples["default"]))
        
        # Build examples string
        examples_str = ""
        for i, ex in enumerate(examples[:num_shots], 1):
            examples_str += f"{i}. Toxic: {ex['toxic']}\n   Neutral: {ex['neutral']}\n"
        
        prompt = PROMPT_TEMPLATE.format(
            examples=examples_str,
            language=language,
            toxic_text=toxic_text,
        )
        
        return prompt

    def detoxify_single(
        self, 
        toxic_text: str, 
        lang: str = "en",
        num_shots: int = 3,
        max_new_tokens: int = 128,
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
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the neutral text (after the prompt)
        if "Neutral text:" in generated:
            neutral = generated.split("Neutral text:")[-1].strip()
        else:
            neutral = generated[len(prompt):].strip()

        # Clean up
        if "\n\n" in neutral:
            neutral = neutral.split("\n\n")[0].strip()

        return neutral

    def detoxify_batch(
        self, 
        texts: List[str], 
        langs: List[str], 
        batch_size: int = 4, 
        device_id: int = 0,
        num_shots: int = 3,
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
    use_4bit: bool = False,
    use_8bit: bool = False,
    num_shots: int = 3,
    temperature: float = 0.7,
) -> pd.DataFrame:
    """Worker function to process a chunk of data on a specific GPU."""
    
    detoxifier = QwenDetoxifier(
        device_id=device_id,
        use_4bit=use_4bit,
        use_8bit=use_8bit,
    )
    
    neutral_sentences = detoxifier.detoxify_batch(
        texts=df_chunk["toxic_sentence"].tolist(),
        langs=df_chunk["lang"].tolist(),
        batch_size=batch_size,
        device_id=device_id,
        num_shots=num_shots,
        temperature=temperature,
    )
    
    df_chunk = df_chunk.copy()
    df_chunk["neutral_sentence"] = neutral_sentences
    return df_chunk


def process_file(
    input_path: str, 
    output_path: str, 
    batch_size: int = 4,
    use_4bit: bool = False,
    use_8bit: bool = False,
    num_shots: int = 3,
    temperature: float = 0.7,
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
                    use_4bit, use_8bit, num_shots, temperature
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
            num_shots=num_shots, temperature=temperature
        )

    final_df.to_csv(output_path, sep="\t", index=False)


def main():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Qwen2.5-7B Few-Shot Detoxification (Multi-GPU)")
    
    parser.add_argument(
        "--input_path",
        type=str,
        default=str(Path(INPUT_DATA_PATH, "dev_inputs.tsv")),
        help="Path to input TSV file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(Path(OUTPUT_DATA_PATH, "qwen_fewshot_output.tsv")),
        help="Path to output TSV file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size per GPU for processing (default: 4)",
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Use 4-bit quantization (~8GB VRAM per GPU)",
    )
    parser.add_argument(
        "--use_8bit",
        action="store_true",
        help="Use 8-bit quantization (~12GB VRAM per GPU)",
    )
    parser.add_argument(
        "--num_shots",
        type=int,
        default=3,
        help="Number of few-shot examples (default: 3)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature (default: 0.7)",
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Qwen2.5-7B Few-Shot Detoxification (Multi-GPU)")
    print("=" * 60)
    process_file(
        args.input_path, args.output_path, args.batch_size,
        args.use_4bit, args.use_8bit, args.num_shots, args.temperature
    )
    print(f"\nProcessing complete. Results saved to {output_path}")


if __name__ == "__main__":
    main()
