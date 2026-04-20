#!/usr/bin/env python3
"""
Qwen2.5-7B Few-Shot Learning for Multilingual Text Detoxification (Multi-GPU)
==============================================================================

Optimized for Kaggle's 2x T4 GPUs (each ~16GB VRAM).
Each GPU processes half the data in parallel.

Features:
- Parallel inference on 2 GPUs (data parallelism)
- Few-shot learning with retrieved examples
- Chain-of-Thought (CoT) prompting
- Multi-language support with language-specific examples
- Retrieval-based example selection using FAISS
- 4-bit quantization for memory efficiency

Usage on Kaggle (2x T4):
    # Basic inference (parallel on 2 GPUs)
    python baseline_qwen_fewshot_multigpu.py --input_path data/test.tsv

    # With retrieval-based few-shot
    python baseline_qwen_fewshot_multigpu.py --use_retrieval --num_shots 3

    # With 4-bit quantization (fits in ~8GB VRAM per GPU)
    python baseline_qwen_fewshot_multigpu.py --use_4bit

Requirements:
    pip install transformers accelerate sentence-transformers faiss-cpu torch
    pip install bitsandbytes  # For 4-bit quantization
"""

import argparse
import json
import os
import sys
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
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
        max_memory: str = "15GB",  # Per GPU memory for Kaggle T4
    ):
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit

        print(f"[GPU {device_id}] Loading model: {model_name}")
        print(f"[GPU {device_id}] Device: {self.device}")
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
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if not (use_4bit or use_8bit) else None,
            "device_map": {"": device_id},  # Force to specific GPU
        }

        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["max_memory"] = {device_id: max_memory}

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

        print(f"[GPU {device_id}] Model loaded successfully!")

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
        ).to(self.device)

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

        for i in tqdm(range(0, len(texts), batch_size), desc=f"GPU {self.device.index}"):
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
# Multi-GPU Parallel Processing
# =============================================================================

def worker_process(
    gpu_id: int,
    texts: List[str],
    langs: List[str],
    args_dict: dict,
    result_queue: mp.Queue,
):
    """Worker process for a single GPU."""
    
    # Set device
    torch.cuda.set_device(gpu_id)
    
    # Initialize model
    detoxifier = QwenDetoxifier(
        model_name=args_dict['model_name'],
        device_id=gpu_id,
        use_4bit=args_dict['use_4bit'],
        use_8bit=args_dict['use_8bit'],
        max_memory=args_dict['max_memory'],
    )
    
    # Process texts
    results = detoxifier.detoxify_batch(
        texts=texts,
        langs=langs,
        batch_size=args_dict['batch_size'],
        use_cot=args_dict['use_cot'],
        num_shots=args_dict['num_shots'],
        use_retrieval=args_dict['use_retrieval'],
        max_new_tokens=args_dict['max_new_tokens'],
        temperature=args_dict['temperature'],
    )
    
    # Put results in queue
    result_queue.put((gpu_id, results))


def parallel_inference_mp(
    df: pd.DataFrame,
    args_dict: dict,
    num_gpus: int = 2,
) -> List[str]:
    """
    Run inference in parallel using multiprocessing.
    Each GPU processes a portion of the data.
    """
    
    texts = df['toxic_sentence'].tolist()
    langs = df['lang'].tolist() if 'lang' in df.columns else ['en'] * len(df)
    
    total_samples = len(texts)
    samples_per_gpu = total_samples // num_gpus
    
    print(f"Total samples: {total_samples}")
    print(f"Using {num_gpus} GPUs")
    print(f"Samples per GPU: ~{samples_per_gpu}")
    
    # Split data
    gpu_data = []
    for i in range(num_gpus):
        start_idx = i * samples_per_gpu
        end_idx = start_idx + samples_per_gpu if i < num_gpus - 1 else total_samples
        gpu_data.append({
            'texts': texts[start_idx:end_idx],
            'langs': langs[start_idx:end_idx],
            'start_idx': start_idx,
            'end_idx': end_idx,
        })
        print(f"GPU {i}: {end_idx - start_idx} samples (indices {start_idx}-{end_idx})")
    
    # Create result queue
    result_queue = mp.Queue()
    
    # Start worker processes
    processes = []
    for gpu_id, data in enumerate(gpu_data):
        p = mp.Process(
            target=worker_process,
            args=(gpu_id, data['texts'], data['langs'], args_dict, result_queue),
        )
        processes.append(p)
        p.start()
    
    # Collect results
    gpu_results = {}
    for _ in range(num_gpus):
        gpu_id, results = result_queue.get()
        gpu_results[gpu_id] = results
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    # Combine results in correct order
    all_results = []
    for gpu_id in range(num_gpus):
        all_results.extend(gpu_results[gpu_id])
    
    return all_results


# =============================================================================
# Alternative: Using torchrun for DDP-based parallelism
# =============================================================================

def setup_distributed():
    """Initialize distributed processing."""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend='nccl')

    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed processing."""
    if dist.is_initialized():
        dist.destroy_process_group()


def ddp_inference_mode(args, rank: int, world_size: int):
    """Inference mode using DDP (torchrun)."""
    
    local_rank = rank
    
    print(f"[Rank {rank}] Starting DDP inference...")
    
    # Initialize model
    detoxifier = QwenDetoxifier(
        model_name=args.model_name,
        device_id=local_rank,
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit,
        max_memory=args.max_memory,
    )
    
    # Load input data
    df = pd.read_csv(args.input_path, sep='\t')
    
    if rank == 0:
        print(f"Loaded {len(df)} samples from {args.input_path}")
    
    # Split data across GPUs
    total_samples = len(df)
    samples_per_gpu = total_samples // world_size
    start_idx = rank * samples_per_gpu
    end_idx = start_idx + samples_per_gpu if rank < world_size - 1 else total_samples
    
    local_df = df.iloc[start_idx:end_idx].copy()
    texts = local_df['toxic_sentence'].tolist()
    langs = local_df['lang'].tolist() if 'lang' in local_df.columns else ['en'] * len(local_df)
    
    print(f"[Rank {rank}] Processing {len(texts)} samples (indices {start_idx}-{end_idx})")
    
    # Generate
    neutral_sentences = detoxifier.detoxify_batch(
        texts=texts,
        langs=langs,
        batch_size=args.batch_size,
        use_cot=args.use_cot,
        num_shots=args.num_shots,
        use_retrieval=args.use_retrieval,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    
    # Gather results
    if world_size > 1:
        results_list = list(zip(range(start_idx, end_idx), neutral_sentences))
        all_results = [None] * world_size
        dist.all_gather_object(all_results, results_list)
        
        if rank == 0:
            # Combine and sort by index
            combined = []
            for results in all_results:
                combined.extend(results)
            combined.sort(key=lambda x: x[0])
            
            # Save results
            df['neutral_sentence'] = [r[1] for r in combined]
            df.to_csv(args.output_path, sep='\t', index=False)
            print(f"\nResults saved to {args.output_path}")
    else:
        df['neutral_sentence'] = neutral_sentences
        df.to_csv(args.output_path, sep='\t', index=False)
        print(f"\nResults saved to {args.output_path}")


# =============================================================================
# Main Functions
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Qwen2.5-7B Few-Shot Text Detoxification (Multi-GPU)"
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name or path",
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
        "--max_memory",
        type=str,
        default="15GB",
        help="Max GPU memory allocation per GPU",
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

    # Multi-GPU arguments
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=2,
        help="Number of GPUs to use (for multiprocessing mode)",
    )
    parser.add_argument(
        "--use_ddp",
        action="store_true",
        help="Use DDP mode (requires torchrun)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Qwen2.5-7B Few-Shot Text Detoxification (Multi-GPU)")
    print("=" * 60)

    # Create output directory
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    # Check available GPUs
    num_available_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_available_gpus}")
    
    if num_available_gpus < args.num_gpus:
        print(f"Warning: Requested {args.num_gpus} GPUs but only {num_available_gpus} available")
        args.num_gpus = num_available_gpus

    if args.use_ddp or 'RANK' in os.environ:
        # DDP mode (use with torchrun)
        rank, world_size, local_rank = setup_distributed()
        print(f"DDP Mode: rank={rank}, world_size={world_size}")
        
        try:
            ddp_inference_mode(args, rank, world_size)
        finally:
            cleanup_distributed()
    else:
        # Multiprocessing mode
        print(f"Multiprocessing Mode: using {args.num_gpus} GPUs")
        
        # Load input data
        df = pd.read_csv(args.input_path, sep='\t')
        print(f"Loaded {len(df)} samples from {args.input_path}")
        
        # Convert args to dict for multiprocessing
        args_dict = {
            'model_name': args.model_name,
            'use_4bit': args.use_4bit,
            'use_8bit': args.use_8bit,
            'max_memory': args.max_memory,
            'batch_size': args.batch_size,
            'use_cot': args.use_cot,
            'num_shots': args.num_shots,
            'use_retrieval': args.use_retrieval,
            'max_new_tokens': args.max_new_tokens,
            'temperature': args.temperature,
        }
        
        # Run parallel inference
        neutral_sentences = parallel_inference_mp(df, args_dict, args.num_gpus)
        
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
    # Required for multiprocessing with CUDA
    mp.set_start_method('spawn', force=True)
    main()
