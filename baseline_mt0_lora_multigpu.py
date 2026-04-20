#!/usr/bin/env python3
"""
mT0-XL + LoRA Fine-tuning for Multilingual Text Detoxification (Multi-GPU)
=============================================================================

Optimized for Kaggle's 2x T4 GPUs (each ~16GB VRAM).

Features:
- Parallel training/inference on 2 GPUs using PyTorch DDP
- LoRA fine-tuning reduces trainable parameters by ~90%
- Supports data augmentation
- Mixed precision training (FP16/BF16)
- Checkpoint saving and resuming

Usage on Kaggle (2x T4):
    # Fine-tune on multilingual detox data (parallel on 2 GPUs)
    torchrun --nproc_per_node=2 baseline_mt0_lora_multigpu.py --mode train --epochs 3 --batch_size 8

    # Inference with fine-tuned model (parallel on 2 GPUs)
    torchrun --nproc_per_node=2 baseline_mt0_lora_multigpu.py --mode inference --input_path data/test.tsv

    # Alternative: Use accelerate launch
    accelerate launch --num_processes 2 baseline_mt0_lora_multigpu.py --mode train

Requirements:
    pip install transformers peft accelerate datasets sentencepiece torch
    pip install bitsandbytes  # Optional: for quantization
"""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional
import json

import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# =============================================================================
# Configuration
# =============================================================================

FILE_PATH = Path(__file__).resolve()
INPUT_DATA_PATH = Path(FILE_PATH.parent, "input_data")
OUTPUT_DATA_PATH = Path(FILE_PATH.parent, "output_data")
CHECKPOINT_PATH = Path(FILE_PATH.parent, "checkpoints")

# Language-specific prompts for detoxification
LANG_PROMPTS = {
    "zh": "排毒：", "es": "Desintoxicar: ", "ru": "Детоксифицируй: ",
    "ar": "إزالة السموم: ", "hi": "विषहरण: ", "uk": "Детоксифікуй: ",
    "de": "Entgiften: ", "am": "መርዝ መርዝ: ", "en": "Detoxify: ",
    "it": "Disintossicare: ", "ja": "解毒: ", "he": "לְסַלֵק רַעַל: ",
    "fr": "Désintoxiquer:", "tt": "Токсиннарны чыгару: ", "hin": "Detoxify: ",
}

LANGUAGES = Literal[
    "zh", "es", "ru", "ar", "hi", "uk", "de", "am",
    "en", "it", "ja", "he", "fr", "tt", "hin",
]


# =============================================================================
# Multi-GPU Utilities
# =============================================================================

def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ:
        # torchrun or accelerate launch
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        # Fallback for single GPU
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend='nccl')

    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """Check if current process is the main process."""
    return rank == 0


def print_on_main(rank: int, message: str):
    """Print only on main process."""
    if is_main_process(rank):
        print(message)


# =============================================================================
# Dataset Classes
# =============================================================================

class DetoxDataset(Dataset):
    """Dataset for fine-tuning detoxification models."""

    def __init__(
        self,
        data_path: Optional[str] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        max_length: int = 256,
        lang_prompts: Dict[str, str] = None,
        data_df: Optional[pd.DataFrame] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lang_prompts = lang_prompts or LANG_PROMPTS

        # Load data
        if data_df is not None:
            self.data = data_df
        elif data_path:
            if data_path.endswith('.tsv'):
                self.data = pd.read_csv(data_path, sep='\t')
            elif data_path.endswith('.csv'):
                self.data = pd.read_csv(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path}")
        else:
            raise ValueError("Either data_path or data_df must be provided")

        # Validate columns
        required_cols = ['toxic_sentence', 'neutral_sentence']
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")

        # Add lang column if missing (default to English)
        if 'lang' not in self.data.columns:
            self.data['lang'] = 'en'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Prepare input with language prompt
        lang = row.get('lang', 'en')
        prompt = self.lang_prompts.get(lang, "Detoxify: ")
        input_text = prompt + row['toxic_sentence']
        target_text = row['neutral_sentence']

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        targets = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze(),
        }


class DetoxInferenceDataset(Dataset):
    """Dataset for inference."""

    def __init__(
        self,
        texts: List[str],
        langs: List[str],
        tokenizer: AutoTokenizer,
        lang_prompts: Dict[str, str] = None,
        max_length: int = 256,
    ):
        self.texts = texts
        self.langs = langs
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lang_prompts = lang_prompts or LANG_PROMPTS

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        lang = self.langs[idx]
        prompt = self.lang_prompts.get(lang, "Detoxify: ")
        text = prompt + self.texts[idx]

        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'original_idx': idx,  # Keep track of original index
        }


# =============================================================================
# LoRA Model Wrapper with Multi-GPU Support
# =============================================================================

class LoRADetoxifier:
    """
    mT0-XL with LoRA fine-tuning for text detoxification.
    Supports multi-GPU training and inference.
    """

    def __init__(
        self,
        model_name: str = "s-nlp/mt0-xl-detox-orpo",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        local_rank: int = 0,
        load_in_8bit: bool = False,
        checkpoint_path: Optional[str] = None,
    ):
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        print_on_main(local_rank, f"Loading base model: {model_name}")
        print_on_main(local_rank, f"Device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Load model with optional quantization
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
        }
        
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = {"": self.device}
        else:
            model_kwargs["device_map"] = {"": self.device}

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)

        # Apply LoRA
        if checkpoint_path and os.path.exists(checkpoint_path):
            print_on_main(local_rank, f"Loading LoRA checkpoint from: {checkpoint_path}")
            self.model = PeftModel.from_pretrained(self.model, checkpoint_path)
        else:
            print_on_main(local_rank, "Applying LoRA configuration...")
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q", "v", "k", "o"],  # Attention layers
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_config)

        self.model.print_trainable_parameters()
        self.lang_prompts = LANG_PROMPTS

    def wrap_ddp(self):
        """Wrap model with DDP for distributed training."""
        self.model = DDP(
            self.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=False,
        )

    def train(
        self,
        train_dataset: DetoxDataset,
        val_dataset: Optional[DetoxDataset] = None,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-4,
        warmup_ratio: float = 0.1,
        checkpoint_dir: str = "checkpoints",
        save_every: int = 1,
        rank: int = 0,
        world_size: int = 1,
    ):
        """Fine-tune the model with LoRA using DDP."""

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Distributed sampler for multi-GPU
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        ) if world_size > 1 else None

        # Data loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
        )

        # Optimizer - only optimize LoRA parameters
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )

        # Scheduler
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Training loop
        self.model.train()
        global_step = 0
        best_loss = float('inf')

        print_on_main(rank, f"\nStarting training for {epochs} epochs...")
        print_on_main(rank, f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
        print_on_main(rank, f"World size: {world_size} GPUs")

        for epoch in range(epochs):
            # Set sampler epoch for proper shuffling
            if train_sampler:
                train_sampler.set_epoch(epoch)

            epoch_loss = 0
            
            # Progress bar only on main process
            if is_main_process(rank):
                progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            else:
                progress_bar = train_loader

            for batch in progress_bar:
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Replace padding token id with -100 for loss calculation
                labels[labels == self.tokenizer.pad_token_id] = -100

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                global_step += 1

                if is_main_process(rank):
                    progress_bar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                    })

            # Synchronize and compute average loss across GPUs
            avg_loss = epoch_loss / len(train_loader)
            if world_size > 1:
                avg_loss_tensor = torch.tensor([avg_loss], device=self.device)
                dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
                avg_loss = (avg_loss_tensor / world_size).item()

            print_on_main(rank, f"\nEpoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

            # Validation (only on main process)
            if val_dataset and is_main_process(rank):
                val_loss = self.validate(val_dataset, batch_size)
                print_on_main(rank, f"Validation Loss: {val_loss:.4f}")

            # Save checkpoint (only on main process)
            if is_main_process(rank) and (epoch + 1) % save_every == 0:
                checkpoint_name = f"checkpoint_epoch_{epoch + 1}"
                checkpoint_full_path = Path(checkpoint_dir) / checkpoint_name
                
                # Get unwrapped model for saving
                model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                model_to_save.save_pretrained(checkpoint_full_path)
                self.tokenizer.save_pretrained(checkpoint_full_path)
                print_on_main(rank, f"Checkpoint saved: {checkpoint_full_path}")

                # Save best model
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_path = Path(checkpoint_dir) / "best_model"
                    model_to_save.save_pretrained(best_path)
                    self.tokenizer.save_pretrained(best_path)
                    print_on_main(rank, f"Best model saved with loss: {best_loss:.4f}")

        print_on_main(rank, "\nTraining complete!")

    def validate(self, val_dataset: DetoxDataset, batch_size: int = 8):
        """Validate the model."""
        self.model.eval()
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                labels[labels == self.tokenizer.pad_token_id] = -100

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                total_loss += outputs.loss.item()

        self.model.train()
        return total_loss / len(val_loader)

    def detoxify_batch(
        self,
        texts: List[str],
        langs: List[str],
        batch_size: int = 8,
        num_beams: int = 5,
    ) -> List[str]:
        """Generate detoxified text for a batch of inputs."""
        self.model.eval()

        dataset = DetoxInferenceDataset(texts, langs, self.tokenizer, self.lang_prompts)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        results = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=128,
                    num_beams=num_beams,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.2,
                    num_return_sequences=1,
                    early_stopping=True,
                )

                decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                results.extend(decoded)

        return results


# =============================================================================
# Data Loading Utilities
# =============================================================================

def load_training_data(
    use_augmentation: bool = False,
    augmentation_factor: int = 2,
):
    """Load and optionally augment training data."""

    print("Loading multilingual_paradetox dataset...")
    dataset = load_dataset("textdetox/multilingual_paradetox")

    # Convert to DataFrame
    all_data = []
    for split in dataset:
        for item in dataset[split]:
            all_data.append({
                'toxic_sentence': item['toxic_sentence'],
                'neutral_sentence': item['neutral_sentence'],
                'lang': item.get('lang', 'en'),
            })

    df = pd.DataFrame(all_data)
    print(f"Loaded {len(df)} training samples")

    if use_augmentation:
        print(f"Applying data augmentation (factor: {augmentation_factor})...")
        # Simple augmentation: duplicate with slight variations
        augmented = df.copy()
        for _ in range(augmentation_factor - 1):
            aug_df = df.copy()
            # Add noise to text (simple example)
            # In practice, use backtranslation or synonym replacement
            augmented = pd.concat([augmented, aug_df], ignore_index=True)
        df = augmented
        print(f"Augmented to {len(df)} samples")

    return df


def split_data(df: pd.DataFrame, val_ratio: float = 0.1):
    """Split data into train and validation sets."""
    # Stratified split by language
    train_dfs = []
    val_dfs = []

    for lang in df['lang'].unique():
        lang_df = df[df['lang'] == lang]
        n_val = int(len(lang_df) * val_ratio)
        
        val_dfs.append(lang_df.iloc[:n_val])
        train_dfs.append(lang_df.iloc[n_val:])

    train_df = pd.concat(train_dfs, ignore_index=True).sample(frac=1, random_state=42)
    val_df = pd.concat(val_dfs, ignore_index=True)

    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    return train_df, val_df


# =============================================================================
# Parallel Inference
# =============================================================================

def parallel_inference(
    detoxifier: LoRADetoxifier,
    df: pd.DataFrame,
    batch_size: int,
    num_beams: int,
    rank: int,
    world_size: int,
    output_path: str,
):
    """Run inference in parallel across GPUs."""
    
    # Split data across GPUs
    total_samples = len(df)
    samples_per_gpu = total_samples // world_size
    start_idx = rank * samples_per_gpu
    end_idx = start_idx + samples_per_gpu if rank < world_size - 1 else total_samples
    
    local_df = df.iloc[start_idx:end_idx].copy()
    
    print_on_main(rank, f"GPU {rank}: Processing {len(local_df)} samples (indices {start_idx}-{end_idx})")
    
    # Generate
    neutral_sentences = detoxifier.detoxify_batch(
        texts=local_df['toxic_sentence'].tolist(),
        langs=local_df['lang'].tolist() if 'lang' in local_df.columns else ['en'] * len(local_df),
        batch_size=batch_size,
        num_beams=num_beams,
    )
    
    local_df['neutral_sentence'] = neutral_sentences
    
    # Gather results on main process
    if world_size > 1:
        # Convert to list of dicts for gather
        results_list = local_df.to_dict('records')
        
        # Gather all results
        all_results = [None] * world_size
        dist.all_gather_object(all_results, results_list)
        
        if is_main_process(rank):
            # Combine all results
            combined_results = []
            for results in all_results:
                combined_results.extend(results)
            final_df = pd.DataFrame(combined_results)
            final_df.to_csv(output_path, sep='\t', index=False)
            print(f"Results saved to {output_path}")
    else:
        local_df.to_csv(output_path, sep='\t', index=False)
        print(f"Results saved to {output_path}")


# =============================================================================
# Main Functions
# =============================================================================

def train_mode(args, rank: int, world_size: int):
    """Training mode."""
    print_on_main(rank, "=" * 60)
    print_on_main(rank, "mT0-XL + LoRA Fine-tuning for Text Detoxification (Multi-GPU)")
    print_on_main(rank, "=" * 60)

    # Load data (only on main process first, then broadcast)
    if is_main_process(rank):
        df = load_training_data(
            use_augmentation=args.use_augmentation,
            augmentation_factor=args.augmentation_factor,
        )
        train_df, val_df = split_data(df, val_ratio=0.1)
    else:
        train_df, val_df = None, None
    
    # Broadcast data to all processes
    if world_size > 1:
        train_df = [train_df]
        val_df = [val_df]
        dist.broadcast_object_list(train_df, src=0)
        dist.broadcast_object_list(val_df, src=0)
        train_df = train_df[0]
        val_df = val_df[0]

    # Initialize model
    detoxifier = LoRADetoxifier(
        model_name=args.model_name,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        local_rank=rank,
        load_in_8bit=args.load_in_8bit,
        checkpoint_path=args.checkpoint_path,
    )

    # Wrap with DDP
    if world_size > 1:
        detoxifier.wrap_ddp()

    # Create datasets
    train_dataset = DetoxDataset(data_df=train_df, tokenizer=detoxifier.tokenizer)
    val_dataset = DetoxDataset(data_df=val_df, tokenizer=detoxifier.tokenizer)

    # Train
    detoxifier.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        rank=rank,
        world_size=world_size,
    )


def inference_mode(args, rank: int, world_size: int):
    """Inference mode."""
    print_on_main(rank, "=" * 60)
    print_on_main(rank, "mT0-XL + LoRA Inference (Multi-GPU)")
    print_on_main(rank, "=" * 60)

    # Initialize model with checkpoint
    detoxifier = LoRADetoxifier(
        model_name=args.model_name,
        local_rank=rank,
        checkpoint_path=args.checkpoint_path,
    )

    # Load input data
    df = pd.read_csv(args.input_path, sep='\t')
    print_on_main(rank, f"Loaded {len(df)} samples from {args.input_path}")

    # Run parallel inference
    parallel_inference(
        detoxifier=detoxifier,
        df=df,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        rank=rank,
        world_size=world_size,
        output_path=args.output_path,
    )


def main():
    parser = argparse.ArgumentParser(
        description="mT0-XL + LoRA Fine-tuning for Text Detoxification (Multi-GPU)"
    )

    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "inference"],
        default="train",
        help="Operation mode",
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="s-nlp/mt0-xl-detox-orpo",
        help="Base model name or path",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to LoRA checkpoint for loading",
    )

    # LoRA arguments
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/mt0_lora",
        help="Directory to save checkpoints",
    )
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")

    # Data arguments
    parser.add_argument(
        "--use_augmentation",
        action="store_true",
        help="Use data augmentation",
    )
    parser.add_argument(
        "--augmentation_factor",
        type=int,
        default=2,
        help="Data augmentation factor",
    )

    # Inference arguments
    parser.add_argument(
        "--input_path",
        type=str,
        default=str(Path(INPUT_DATA_PATH, "dev_inputs.tsv")),
        help="Input TSV file path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(Path(OUTPUT_DATA_PATH, "mt0_lora_output.tsv")),
        help="Output TSV file path",
    )
    parser.add_argument("--num_beams", type=int, default=5, help="Number of beams for generation")

    args = parser.parse_args()

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    
    print_on_main(rank, f"Distributed setup: rank={rank}, world_size={world_size}, local_rank={local_rank}")

    # Create directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        if args.mode == "train":
            train_mode(args, local_rank, world_size)
        else:
            inference_mode(args, local_rank, world_size)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
