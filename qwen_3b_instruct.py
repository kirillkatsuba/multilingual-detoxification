import argparse
import concurrent.futures
import math
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import torch.multiprocessing as mp
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

FILE_PATH: str = Path(__file__).resolve()
# Following your specific directory structure
INPUT_DATA_PATH: str = Path(FILE_PATH.parent.parent.parent, "input_data")
OUTPUT_DATA_PATH: str = Path(FILE_PATH.parent.parent.parent, "output_data")


class QwenDetoxifier:
    """
    A class for detoxifying text using Qwen2.5-3B-Instruct.
    Optimized for speed (Greedy Search) and reproducibility.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", device_id: int = 0):
        """
        Initialize the QwenDetoxifier for a specific GPU.
        """
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

        # Load model in bfloat16 (Modern GPUs like T4/3090 handle this perfectly)
        # 3B parameters fit easily into 15GB VRAM without quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
            trust_remote_code=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Decoder models (Qwen/Llama) MUST use left-padding for batch generation
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _prepare_batch(self, batch_texts: List[str]) -> Dict[str, torch.Tensor]:
        """Format prompts into ChatML and tokenize."""
        
        # System prompt ensures the model behaves like a detoxifier and stays reproducible
        formatted_prompts = [
            f"<|im_start|>system\nYou are a helpful assistant. Rewrite the following text to be non-toxic and neutral while preserving the original meaning. Output ONLY the rewritten text and nothing else.<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
            for text in batch_texts
        ]
        
        return self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

    def _generate_batch(self, encodings: Dict[str, torch.Tensor]) -> List[str]:
        """Generate detoxified text. Uses Greedy Search for speed and reproducibility."""
        with torch.no_grad():
            outputs = self.model.generate(
                **encodings,
                max_new_tokens=128,
                do_sample=False,  # Greedy search (Same output every time)
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Extract only the newly generated text (the part after the prompt)
        prompt_length = encodings.input_ids.shape[-1]
        new_tokens = outputs[:, prompt_length:]
        
        return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    def detoxify_batch(
        self, texts: List[str], batch_size: int = 8, device_id: int = 0
    ) -> List[str]:
        """Process the text in batches."""
        results = []
        
        for i in tqdm(
            range(0, len(texts), batch_size), 
            desc=f"GPU {device_id} Processing",
            position=device_id,
            leave=True
        ):
            current_batch = texts[i : i + batch_size]
            encodings = self._prepare_batch(current_batch)
            batch_results = self._generate_batch(encodings)
            results.extend([res.strip() for res in batch_results])

        return results


def process_chunk(df_chunk: pd.DataFrame, device_id: int, batch_size: int) -> pd.DataFrame:
    """Worker function to process a chunk of data on a specific GPU."""
    detoxifier = QwenDetoxifier(device_id=device_id)
    
    neutral_sentences = detoxifier.detoxify_batch(
        texts=df_chunk["toxic_sentence"].tolist(),
        batch_size=batch_size,
        device_id=device_id
    )
    
    df_chunk = df_chunk.copy()
    df_chunk["neutral_sentence"] = neutral_sentences
    return df_chunk


def process_file(input_path: str, output_path: str, batch_size: int = 8):
    """
    Process an input TSV file, split across multiple GPUs, and save output.
    """
    df = pd.read_csv(input_path, sep="\t")
    
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(f"Total rows: {len(df)} | GPUs: {num_gpus} | Batch Size: {batch_size}")

    if num_gpus > 1:
        chunk_size = math.ceil(len(df) / num_gpus)
        chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
        
        results = []
        ctx = mp.get_context('spawn') 
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_gpus, mp_context=ctx) as executor:
            futures = [
                executor.submit(process_chunk, chunk, gpu_id, batch_size) 
                for gpu_id, chunk in enumerate(chunks)
            ]
            
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                
        final_df = pd.concat(results).sort_index()
    else:
        final_df = process_chunk(df, device_id=0, batch_size=batch_size)

    final_df.to_csv(output_path, sep="\t", index=False)


def main():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Multi-GPU Qwen-3B Detoxification")
    
    parser.add_argument(
        "--input_path",
        type=str,
        default=str(Path(INPUT_DATA_PATH, "dev_inputs.tsv")),
        help="Path to input TSV file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(Path(OUTPUT_DATA_PATH, "qwen_3b_output.tsv")),
        help="Path to output TSV file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8, # Qwen-3B is efficient; Batch 8 fits well on T4
        help="Batch size per GPU",
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Starting Qwen-3B Multi-GPU Processing...")
    process_file(args.input_path, args.output_path, args.batch_size)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()