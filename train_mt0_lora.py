"""
mT0-XL + LoRA Fine-tuning Training Script (Multi-GPU)
======================================================

Training script for fine-tuning mT0-XL with LoRA adapters.
Uses multiprocessing with ProcessPoolExecutor for training on multiple GPUs.

Usage:
    # Train on 2 GPUs
    python train_mt0_lora_2gpus.py --epochs 3 --batch_size 8 --output_dir ./checkpoints/mt0_lora

    # Train with specific learning rate
    python train_mt0_lora_2gpus.py --epochs 3 --learning_rate 5e-4 --lora_rank 16

    # Resume from checkpoint
    python train_mt0_lora_2gpus.py --resume_from ./checkpoints/mt0_lora/checkpoint_epoch_2

Requirements:
    pip install transformers peft torch pandas tqdm datasets
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.multiprocessing as mp
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

FILE_PATH: str = Path(__file__).resolve()
INPUT_DATA_PATH: str = Path(FILE_PATH.parent, "input_data")
OUTPUT_DATA_PATH: str = Path(FILE_PATH.parent, "output_data")
CHECKPOINT_PATH: str = Path(FILE_PATH.parent, "checkpoints")

# Language-specific prompts
LANG_PROMPTS = {
    "zh": "排毒：", "es": "Desintoxicar: ", "ru": "Детоксифицируй: ",
    "ar": "إزالة السموم: ", "hi": "विषहरण: ", "uk": "Детоксифікуй: ",
    "de": "Entgiften: ", "am": "መርዝ መርዝ: ", "en": "Detoxify: ",
    "it": "Disintossicare: ", "ja": "解毒: ", "he": "לְסַלֵק רַעַל: ",
    "fr": "Désintoxiquer:", "tt": "Токсиннарны чыгару: ", "hin": "Detoxify: ",
}


class DetoxDataset(Dataset):
    """Dataset for fine-tuning detoxification models."""

    def __init__(
        self,
        data_df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int = 256,
        lang_prompts: Dict[str, str] = None,
    ):
        self.data = data_df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lang_prompts = lang_prompts or LANG_PROMPTS

        # Validate columns
        required_cols = ['toxic_sentence', 'neutral_sentence']
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")

        # Add lang column if missing
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


class MT0LoRATrainer:
    """
    mT0-XL LoRA Fine-tuning Trainer.
    Trains on a single GPU with support for checkpoint saving.
    """

    def __init__(
        self,
        model_name: str = "s-nlp/mt0-xl-detox-orpo",
        device_id: int = 0,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        resume_from: Optional[str] = None,
        output_dir: str = "checkpoints/mt0_lora",
    ):
        """Initialize the trainer for a specific GPU."""
        
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device_id}")
        else:
            self.device = torch.device("cpu")

        self.device_id = device_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[GPU {device_id}] Loading base model: {model_name}")
        print(f"[GPU {device_id}] Device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Load base model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

        # Apply LoRA or load from checkpoint
        if resume_from and Path(resume_from).exists():
            print(f"[GPU {device_id}] Loading LoRA checkpoint from: {resume_from}")
            self.model = PeftModel.from_pretrained(self.model, resume_from)
        else:
            print(f"[GPU {device_id}] Applying LoRA configuration...")
            print(f"[GPU {device_id}] LoRA rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}")
            
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q", "v", "k", "o"],  # Attention layers
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_config)

        self.model = self.model.to(self.device)
        self.model.print_trainable_parameters()
        
        print(f"[GPU {device_id}] Model ready for training!")

    def train(
        self,
        train_dataset: DetoxDataset,
        val_dataset: Optional[DetoxDataset] = None,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-4,
        warmup_ratio: float = 0.1,
        save_every: int = 1,
    ):
        """Train the model with LoRA fine-tuning."""

        # Data loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
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

        print(f"\n[GPU {self.device_id}] Starting training...")
        print(f"[GPU {self.device_id}] Total steps: {total_steps}, Warmup steps: {warmup_steps}")
        print(f"[GPU {self.device_id}] Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")

        for epoch in range(epochs):
            epoch_loss = 0
            progress_bar = tqdm(
                train_loader, 
                desc=f"[GPU {self.device_id}] Epoch {epoch + 1}/{epochs}",
                position=self.device_id,
                leave=True
            )

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

                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })

            avg_loss = epoch_loss / len(train_loader)
            print(f"\n[GPU {self.device_id}] Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

            # Validation
            if val_dataset:
                val_loss = self.validate(val_dataset, batch_size)
                print(f"[GPU {self.device_id}] Validation Loss: {val_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                checkpoint_name = f"checkpoint_epoch_{epoch + 1}"
                checkpoint_path = self.output_dir / checkpoint_name
                self.model.save_pretrained(checkpoint_path)
                self.tokenizer.save_pretrained(checkpoint_path)
                print(f"[GPU {self.device_id}] Checkpoint saved: {checkpoint_path}")

                # Save best model
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_path = self.output_dir / "best_model"
                    self.model.save_pretrained(best_path)
                    self.tokenizer.save_pretrained(best_path)
                    print(f"[GPU {self.device_id}] Best model saved with loss: {best_loss:.4f}")

        print(f"\n[GPU {self.device_id}] Training complete!")
        
        # Save final model
        final_path = self.output_dir / "final_model"
        self.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        print(f"[GPU {self.device_id}] Final model saved: {final_path}")

    def validate(self, val_dataset: DetoxDataset, batch_size: int = 8) -> float:
        """Validate the model."""
        self.model.eval()
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[GPU {self.device_id}] Validating"):
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


def load_training_data(val_ratio: float = 0.1):
    """Load and split training data."""

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


def train_on_gpu(
    device_id: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    warmup_ratio: float,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    output_dir: str,
    resume_from: Optional[str],
    save_every: int,
):
    """Worker function to train on a specific GPU."""
    
    trainer = MT0LoRATrainer(
        device_id=device_id,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        resume_from=resume_from,
        output_dir=output_dir,
    )

    train_dataset = DetoxDataset(data_df=train_df, tokenizer=trainer.tokenizer)
    val_dataset = DetoxDataset(data_df=val_df, tokenizer=trainer.tokenizer)

    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        save_every=save_every,
    )


def main():
    # Setup multiprocessing
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="mT0-XL + LoRA Fine-tuning Training")
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="s-nlp/mt0-xl-detox-orpo",
        help="Base model name or path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(CHECKPOINT_PATH, "mt0_lora")),
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    # LoRA arguments
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")

    # GPU arguments
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="GPU device ID to use for training (default: 0)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("mT0-XL + LoRA Fine-tuning Training")
    print("=" * 60)

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    train_df, val_df = load_training_data(val_ratio=args.val_ratio)

    # Run training on specified GPU
    train_on_gpu(
        device_id=args.device_id,
        train_df=train_df,
        val_df=val_df,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()