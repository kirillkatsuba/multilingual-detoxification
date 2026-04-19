import argparse
import concurrent.futures
import math
from pathlib import Path
from typing import Dict, List, Literal

import pandas as pd
import torch
import torch.multiprocessing as mp
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

FILE_PATH: str = Path(__file__).resolve()
INPUT_DATA_PATH: str = Path(FILE_PATH.parent.parent.parent, "input_data")
OUTPUT_DATA_PATH: str = Path(FILE_PATH.parent.parent.parent, "output_data")


class MT0Detoxifier:
    """
    A class for detoxifying text using the MT0 model with language-specific prompts.
    Modified to support specific GPU allocation and memory-efficient bfloat16 loading.
    """

    LANGUAGES = Literal[
        "zh", "es", "ru", "ar", "hi", "uk", "de", "am", 
        "en", "it", "ja", "he", "fr", "tt", "hin",
    ]

    def __init__(self, model_name: str = "s-nlp/mt0-xl-detox-orpo", device_id: int = 0):
        """
        Initialize the MT0Detoxifier for a specific GPU.
        """
        # Assign to specific GPU
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device_id}")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Load model in bfloat16 to halve VRAM usage (fits easily on 15GB GPUs)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        self.lang_prompts = {
            "zh": "排毒：", "es": "Desintoxicar: ", "ru": "Детоксифицируй: ",
            "ar": "إزالة السموم: ", "hi": "विषहरण: ", "uk": "Детоксифікуй: ",
            "de": "Entgiften: ", "am": "መርዝ መርዝ: ", "en": "Detoxify: ",
            "it": "Disintossicare: ", "ja": "解毒: ", "he": "לְסַלֵק רַעַל: ",
            "fr": "Désintoxiquer:", "tt": "Токсиннарны чыгару: ", "hin": "Detoxify: ",
        }

    def _prepare_batch(self, batch_texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of texts and prepare for model input."""
        return self.tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.device)

    def _generate_batch(self, encodings: Dict[str, torch.Tensor]) -> List[str]:
        """Generate detoxified text for a batch of inputs."""
        with torch.no_grad():
            outputs = self.model.generate(
                **encodings,
                max_length=128,
                num_beams=5,             # Reduced from 10 to save memory and speed up
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                num_beam_groups=5,
                diversity_penalty=2.5,
                num_return_sequences=1,
                early_stopping=True,
                trust_remote_code=True,  # Required for group-beam-search
            )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def detoxify_batch(
        self, texts: List[str], langs: List[LANGUAGES], batch_size: int = 4, device_id: int = 0
    ) -> List[str]:
        """Detoxify a batch of texts with their corresponding languages."""
        
        batch_texts = [
            self.lang_prompts.get(lang, "Detoxify: ") + text
            for text, lang in zip(texts, langs)
        ]

        results = []
        
        # position=device_id prevents tqdm output from scrambling on multiple GPUs
        for i in tqdm(
            range(0, len(batch_texts), batch_size), 
            desc=f"GPU {device_id} Processing",
            position=device_id,
            leave=True
        ):
            current_batch = batch_texts[i : i + batch_size]
            encodings = self._prepare_batch(current_batch)
            batch_results = self._generate_batch(encodings)
            results.extend(batch_results)

        return results


# Worker function MUST be at the top level to work with Python multiprocessing
def process_chunk(df_chunk: pd.DataFrame, device_id: int, batch_size: int) -> pd.DataFrame:
    """Worker function to process a chunk of data on a specific GPU."""
    
    detoxifier = MT0Detoxifier(device_id=device_id)
    
    neutral_sentences = detoxifier.detoxify_batch(
        texts=df_chunk["toxic_sentence"].tolist(),
        langs=df_chunk["lang"].tolist(),
        batch_size=batch_size,
        device_id=device_id
    )
    
    # Create a copy to avoid SettingWithCopyWarning
    df_chunk = df_chunk.copy()
    df_chunk["neutral_sentence"] = neutral_sentences
    return df_chunk


def process_file(input_path: str, output_path: str, batch_size: int = 4):
    """
    Process an input TSV file, split across multiple GPUs, and save output.
    """
    df = pd.read_csv(input_path, sep="\t")
    
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(f"Total rows to process: {len(df)}")
    print(f"Found {num_gpus} device(s) available for processing.")

    if num_gpus > 1:
        # Split dataframe into equal chunks for each GPU
        chunk_size = math.ceil(len(df) / num_gpus)
        chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
        
        results = []
        # Use spawn method to avoid CUDA initialization errors in multiprocessing
        ctx = mp.get_context('spawn') 
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_gpus, mp_context=ctx) as executor:
            futures = [
                executor.submit(process_chunk, chunk, gpu_id, batch_size) 
                for gpu_id, chunk in enumerate(chunks)
            ]
            
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                
        # Recombine and sort by original index
        final_df = pd.concat(results).sort_index()
    else:
        # Fallback to single GPU/CPU
        final_df = process_chunk(df, device_id=0, batch_size=batch_size)

    # Save results
    final_df.to_csv(output_path, sep="\t", index=False)


def main():
    # Setup multiprocessing safe start (Critical for PyTorch multi-GPU processing)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Detoxify text using MT0 baseline model (Multi-GPU)")
    
    parser.add_argument(
        "--input_path",
        type=str,
        default=str(Path(INPUT_DATA_PATH, "dev_inputs.tsv")),
        help="Path to input TSV file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(Path(OUTPUT_DATA_PATH, "baseline_mt0_dev.tsv")),
        help="Path to output TSV file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,  # Lowered default batch size to prevent OOM
        help="Batch size per GPU for processing (default: 4)",
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Starting Multi-GPU Processing...")
    process_file(args.input_path, args.output_path, args.batch_size)
    print(f"\nProcessing complete. Results saved to {output_path}")


if __name__ == "__main__":
    main()