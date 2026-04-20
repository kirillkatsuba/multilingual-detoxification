#!/usr/bin/env python3
"""
Qwen2.5-7B Few-Shot Learning for Multilingual Text Detoxification
==================================================================

This script uses Qwen2.5-7B-Instruct with few-shot prompting for text detoxification.
Optimized for 32GB GPU with optional 4-bit quantization.

Features:
- Few-shot learning with retrieved examples
- Chain-of-Thought (CoT) prompting
- Multi-language support with language-specific examples
- Retrieval-based example selection using FAISS
- 4-bit quantization for memory efficiency

Usage:
    # Basic inference
    python baseline_qwen_fewshot.py --input_path data/test.tsv

    # With retrieval-based few-shot
    python baseline_qwen_fewshot.py --use_retrieval --num_shots 3

    # With 4-bit quantization (fits in ~16GB VRAM)
    python baseline_qwen_fewshot.py --use_4bit

Requirements:
    pip install transformers accelerate sentence-transformers faiss-cpu
    pip install bitsandbytes  # For 4-bit quantization
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

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

# Few-shot examples per language (manually curated high-quality examples)
FEW_SHOT_EXAMPLES = {
    "en": [
        {
            "toxic": "What the f**k is this about?",
            "neutral": "What is this about?",
            "explanation": "Removed the expletive while keeping the question intact."
        },
        {
            "toxic": "He's such a d*mn idiot!",
            "neutral": "He's not very smart.",
            "explanation": "Replaced offensive language with neutral description."
        },
        {
            "toxic": "This is total cr*p!",
            "neutral": "This is not good.",
            "explanation": "Replaced mild profanity with neutral expression."
        },
    ],
    "ru": [
        {
            "toxic": "Какого хр*на тут происходит?",
            "neutral": "Что тут происходит?",
            "explanation": "Убрано ненормативное слово, сохранён смысл вопроса."
        },
        {
            "toxic": "Он такой ид*от!",
            "neutral": "Он не очень умный.",
            "explanation": "Заменено оскорбление на нейтральное описание."
        },
    ],
    "de": [
        {
            "toxic": "Was zur H*lle ist das?",
            "neutral": "Was ist das?",
            "explanation": "Der Fluch wurde entfernt, die Frage beibehalten."
        },
        {
            "toxic": "Das ist totaler Sch*ss!",
            "neutral": "Das ist nicht gut.",
            "explanation": "Umgangssprachlicher Ausdruck durch neutrales ersetzt."
        },
    ],
    "zh": [
        {
            "toxic": "这到底是什么鬼东西？",
            "neutral": "这到底是什么东西？",
            "explanation": "移除了不雅词汇，保留问题原意。"
        },
        {
            "toxic": "真他妈的烦人！",
            "neutral": "真烦人！",
            "explanation": "删除了粗俗表达，保持情绪描述。"
        },
    ],
    "es": [
        {
            "toxic": "¿Qué m*erda es esto?",
            "neutral": "¿Qué es esto?",
            "explanation": "Se eliminó el lenguaje ofensivo manteniendo la pregunta."
        },
    ],
    "fr": [
        {
            "toxic": "C'est quoi cette m*erde?",
            "neutral": "C'est quoi ça?",
            "explanation": "Remplacement du terme vulgaire par une expression neutre."
        },
    ],
    # Default to English examples for languages without specific examples
    "default": [
        {
            "toxic": "This is so d*mn stupid!",
            "neutral": "This is not very good.",
            "explanation": "Replaced offensive language with neutral expression."
        },
    ],
}

# Chain-of-Thought prompt template
COT_PROMPT_TEMPLATE = """You are a text detoxification assistant. Your task is to rewrite toxic text into neutral, non-offensive language while preserving the original meaning as much as possible.

Guidelines:
1. Remove or replace toxic words, profanity, and offensive language
2. Keep the main message and meaning intact
3. Make the text sound natural and fluent
4. Don't change names, places, or factual information

{examples}

Now, detoxify the following {language} text:

Toxic text: {toxic_text}

Step 1: Identify toxic words or phrases.
Step 2: Find neutral alternatives.
Step 3: Rewrite the sentence.

Neutral text:"""

# Simple prompt template (without CoT)
SIMPLE_PROMPT_TEMPLATE = """You are a text detoxification assistant. Rewrite the following toxic text in a neutral, non-offensive way while preserving the meaning.

{examples}

Toxic text ({language}): {toxic_text}

Neutral text:"""


# =============================================================================
# Few-Shot Example Retriever
# =============================================================================

class ExampleRetriever:
    """Retrieve similar examples for few-shot prompting using FAISS."""

    def __init__(self, examples: Dict[str, List[Dict]] = None):
        self.examples = examples or FEW_SHOT_EXAMPLES
        self.encoder = None
        self.index = None
        self.all_examples = []
        self.embeddings = None

    def build_index(self):
        """Build FAISS index for similarity search."""
        if not FAISS_AVAILABLE:
            print("FAISS not available. Using fixed examples.")
            return

        print("Building FAISS index for example retrieval...")
        self.encoder = SentenceTransformer('sentence-transformers/LaBSE')

        # Flatten all examples
        self.all_examples = []
        texts_to_encode = []

        for lang, ex_list in self.examples.items():
            for ex in ex_list:
                self.all_examples.append({**ex, 'lang': lang})
                texts_to_encode.append(ex['toxic'])

        # Encode all toxic sentences
        self.embeddings = self.encoder.encode(texts_to_encode, show_progress_bar=True)
        self.embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)

        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings.astype('float32'))

        print(f"Indexed {len(self.all_examples)} examples")

    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve k most similar examples."""
        if self.index is None or not FAISS_AVAILABLE:
            return []

        # Encode query
        query_emb = self.encoder.encode([query])
        query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)

        # Search
        distances, indices = self.index.search(query_emb.astype('float32'), k)

        results = [self.all_examples[i] for i in indices[0]]
        return results


# =============================================================================
# Qwen Detoxifier
# =============================================================================

class QwenDetoxifier:
    """Qwen2.5-7B-Instruct for text detoxification with few-shot learning."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device_id: int = 0,
        use_4bit: bool = False,
        use_8bit: bool = False,
        use_flash_attention: bool = True,
        max_memory: str = "28GB",
    ):
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit

        print(f"Loading model: {model_name}")
        print(f"Device: {self.device}")
        print(f"Quantization: 4-bit={use_4bit}, 8-bit={use_8bit}")

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
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if not (use_4bit or use_8bit) else None,
            "device_map": "auto",
        }

        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config

        if use_4bit or use_8bit:
            model_kwargs["max_memory"] = {0: max_memory}

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side='left',  # For decoder-only models
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize example retriever
        self.retriever = ExampleRetriever()

        # Language name mapping
        self.lang_names = {
            "en": "English", "ru": "Russian", "de": "German", "zh": "Chinese",
            "es": "Spanish", "fr": "French", "it": "Italian", "ja": "Japanese",
            "ar": "Arabic", "hi": "Hindi", "uk": "Ukrainian", "he": "Hebrew",
            "am": "Amharic", "tt": "Tatar", "hin": "Hinglish",
        }

        print("Model loaded successfully!")

    def build_examples_string(self, examples: List[Dict], lang: str = "en") -> str:
        """Build few-shot examples string for the prompt."""
        if not examples:
            # Use default examples if none provided
            examples = self.retriever.examples.get(lang, self.retriever.examples.get("en", []))

        example_str = "Examples:\n\n"
        for i, ex in enumerate(examples[:3], 1):  # Limit to 3 examples
            example_str += f"{i}. Toxic: {ex['toxic']}\n"
            example_str += f"   Neutral: {ex['neutral']}\n\n"

        return example_str

    def build_prompt(
        self,
        toxic_text: str,
        lang: str = "en",
        use_cot: bool = True,
        num_shots: int = 3,
        retrieved_examples: List[Dict] = None,
    ) -> str:
        """Build the prompt for detoxification."""

        language = self.lang_names.get(lang, lang)

        # Get examples
        if retrieved_examples:
            examples = retrieved_examples[:num_shots]
        else:
            examples = self.retriever.examples.get(lang, self.retriever.examples.get("en", []))

        examples_str = self.build_examples_string(examples, lang)

        # Build prompt
        if use_cot:
            prompt = COT_PROMPT_TEMPLATE.format(
                examples=examples_str,
                language=language,
                toxic_text=toxic_text,
            )
        else:
            prompt = SIMPLE_PROMPT_TEMPLATE.format(
                examples=examples_str,
                language=language,
                toxic_text=toxic_text,
            )

        return prompt

    def detoxify_single(
        self,
        toxic_text: str,
        lang: str = "en",
        use_cot: bool = True,
        num_shots: int = 3,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Detoxify a single text."""

        # Build prompt
        prompt = self.build_prompt(
            toxic_text=toxic_text,
            lang=lang,
            use_cot=use_cot,
            num_shots=num_shots,
        )

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the neutral text (after the prompt)
        if "Neutral text:" in generated:
            neutral = generated.split("Neutral text:")[-1].strip()
        else:
            neutral = generated[len(prompt):].strip()

        # Clean up - remove any additional explanations
        if "\n\n" in neutral:
            neutral = neutral.split("\n\n")[0].strip()
        if "\nStep" in neutral:
            neutral = neutral.split("\nStep")[0].strip()

        return neutral

    def detoxify_batch(
        self,
        texts: List[str],
        langs: List[str],
        batch_size: int = 4,
        use_cot: bool = True,
        num_shots: int = 3,
        use_retrieval: bool = False,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
    ) -> List[str]:
        """Detoxify a batch of texts."""

        # Build retrieval index if needed
        if use_retrieval and FAISS_AVAILABLE and self.retriever.index is None:
            self.retriever.build_index()

        results = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Processing"):
            batch_texts = texts[i:i + batch_size]
            batch_langs = langs[i:i + batch_size]

            batch_results = []

            for text, lang in zip(batch_texts, batch_langs):
                # Get retrieved examples if using retrieval
                retrieved = None
                if use_retrieval and self.retriever.index is not None:
                    retrieved = self.retriever.retrieve(text, k=num_shots)

                # Detoxify
                neutral = self.detoxify_single(
                    toxic_text=text,
                    lang=lang,
                    use_cot=use_cot,
                    num_shots=num_shots,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                batch_results.append(neutral)

            results.extend(batch_results)

        return results


# =============================================================================
# Main Functions
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Qwen2.5-7B Few-Shot Text Detoxification"
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name or path",
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
        help="Use 4-bit quantization (~16GB VRAM)",
    )
    parser.add_argument(
        "--use_8bit",
        action="store_true",
        help="Use 8-bit quantization (~20GB VRAM)",
    )
    parser.add_argument(
        "--max_memory",
        type=str,
        default="28GB",
        help="Max GPU memory allocation",
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
        default=3,
        help="Number of few-shot examples",
    )
    parser.add_argument(
        "--use_retrieval",
        action="store_true",
        help="Use retrieval-based example selection",
    )

    # Generation arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate",
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
        default=str(Path(OUTPUT_DATA_PATH, "qwen_fewshot_output.tsv")),
        help="Output TSV file path",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Qwen2.5-7B Few-Shot Text Detoxification")
    print("=" * 60)

    # Create output directory
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    # Initialize model
    detoxifier = QwenDetoxifier(
        model_name=args.model_name,
        device_id=args.device_id,
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit,
        max_memory=args.max_memory,
    )

    # Load input data
    df = pd.read_csv(args.input_path, sep='\t')
    print(f"Loaded {len(df)} samples from {args.input_path}")

    # Get languages (default to English if not present)
    langs = df['lang'].tolist() if 'lang' in df.columns else ['en'] * len(df)

    # Generate
    neutral_sentences = detoxifier.detoxify_batch(
        texts=df['toxic_sentence'].tolist(),
        langs=langs,
        batch_size=args.batch_size,
        use_cot=args.use_cot,
        num_shots=args.num_shots,
        use_retrieval=args.use_retrieval,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    # Save results
    df['neutral_sentence'] = neutral_sentences
    df.to_csv(args.output_path, sep='\t', index=False)
    print(f"\nResults saved to {args.output_path}")

    # Print sample results
    print("\nSample results:")
    for i in range(min(3, len(df))):
        print(f"\nOriginal: {df['toxic_sentence'].iloc[i]}")
        print(f"Detoxified: {df['neutral_sentence'].iloc[i]}")


if __name__ == "__main__":
    main()
