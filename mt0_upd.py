import argparse
import re
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset

FILE_PATH: str = Path(__file__).resolve()
INPUT_DATA_PATH: str = Path(FILE_PATH.parent.parent.parent, "input_data")
OUTPUT_DATA_PATH: str = Path(FILE_PATH.parent.parent.parent, "output_data")

# ==============================================================================
# UPGRADE 1: Lexical Filter (Replace dummy words with HuggingFace Lexicon later)
# ==============================================================================

def get_toxic_lexicon() -> set:
    """
    Download the official PAN 2025 multilingual toxic lexicon from 
    HuggingFace ('textdetox/multilingual_toxic_lexicon') and load it here.
    """
    # Load dataset with local caching
    ds = load_dataset(
        "textdetox/multilingual_toxic_lexicon",
        split="train",
        trust_remote_code=True
    )
    
    # The dataset typically contains a 'word' column with toxic terms
    # across multiple languages (EN, RU, UK, ES, DE, IT, FR, HE, HI, 
    # JA, TT, AM, AR, ZH, and more)
    column = None
    for col in ("word", "text", "toxic_word", "term"):
        if col in ds.column_names:
            column = col
            break
    
    if column is None:
        # If no known column, use the first string column
        column = ds.column_names[0]
    
    # Build the lexicon set (lowercased and stripped)
    lexicon = set()
    for item in ds[column]:
        if isinstance(item, str) and item.strip():
            lexicon.add(item.strip().lower())
    
    return lexicon


# ==============================================================================
# UPGRADE 2: Strict Multilingual Instructions
# ==============================================================================
LANG_INSTRUCTIONS = {
    "en": "Task: Detoxify the following English text. Rewrite it to be completely neutral, removing all profanity and insults while strictly preserving the original meaning. Reply only in English.",
    "es": "Tarea: Desintoxicar el siguiente texto en español. Reescríbelo para que sea completamente neutral, eliminando todas las malas palabras y los insultos mientras preservas estrictamente el significado original. Responde solo en español.",
    "fr": "Tâche : Désintoxiquer le texte français suivant. Réécrivez-le pour qu'il soit complètement neutre, en supprimant tous les mots grossiers et les insultes tout en préservant strictement le sens original. Répondez uniquement en français.",
    "it": "Compito: Disintossicare il seguente testo in italiano. Riscrivilo in modo che sia completamente neutrale, rimuovendo tutte le volgarità e gli insulti pur preservando rigorosamente il significato originale. Rispondi solo in italiano.",
    "de": "Aufgabe: Entgiften Sie den folgenden deutschen Text. Schreiben Sie ihn völlig neutral um, entfernen Sie alle Obszönitäten und Beleidigungen und behalten Sie dabei die ursprüngliche Bedeutung strikt bei. Antworten Sie nur auf Deutsch.",
    "ru": "Задача: Сделай следующий русский текст нетоксичным. Перепиши его так, чтобы он стал полностью нейтральным, удали всю нецензурную лексику и оскорбления, строго сохранив первоначальный смысл. Отвечай только на русском языке.",
    "uk": "Завдання: Зробіть наступний український текст нетоксичним. Перепишіть його так, щоб він став повністю нейтральним, видаліть усю нецензурну лексику та образи, суворо зберігши початковий зміст. Відповідайте лише українською мовою.",
    "zh": "任务：将以下中文文本进行解毒。 将其改写为完全中性的表达，删除所有脏话和侮辱性词汇，同时严格保留原意。请只用中文回复。",
    "ja": "タスク：以下の日本語のテキストを無毒化してください。 すべての冒とく的な表現や侮辱を取り除き、元の意味を厳密に保ちながら、完全に中立的な表現に書き換えてください。日本語でのみ返信してください。",
    "ar": "مهمة: إزالة السموم من النص العربي التالي. أعد كتابته ليكون محايدًا تمامًا، وقم بإزالة جميع الألفاظ النابية والشتائم مع الحفاظ بدقة على المعنى الأصلي. أجب باللغة العربية فقط.",
    "he": "משימה: נקה את הטקסט הבא בעברית מרעילות. שכתב אותו כך שיהיה ניטרלי לחלוטין, הסר את כל הקללות והעלבונות תוך שמירה קפדנית על המשמעות המקורית. השב בעברית בלבד.",
    "hi": "कार्य: निम्नलिखित हिंदी पाठ को गैर-विषाक्त बनाएं। सभी अपशब्दों और अपमानों को हटाते हुए, मूल अर्थ को सख्ती से बनाए रखते हुए, इसे पूरी तरह से तटस्थ बनाने के लिए फिर से लिखें। केवल हिंदी में उत्तर दें。",
    "hin": "Task: Detoxify the following Hinglish (Hindi + English mix) text. Rewrite it to be completely neutral, removing all profanity and insults while strictly preserving the original meaning. You MUST reply strictly in Hinglish (using Roman script).",
    "am": "Task: Detoxify the following Amharic text. Rewrite it to be completely neutral, removing all profanity and insults while strictly preserving the original meaning. You MUST reply strictly in the Amharic language and Amharic script.",
    "tt": "Task: Detoxify the following Tatar text. Rewrite it to be completely neutral, removing all profanity and insults while strictly preserving the original meaning. You MUST reply strictly in the Tatar language (using Cyrillic script)."
}

class DetoxificationDataset(Dataset):
    """Dataset for batch processing detoxification tasks."""

    def __init__(self, texts: List[str], langs: List[str]):
        self.texts = texts
        self.langs = langs

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx) -> Tuple[str, str]:
        lang = self.langs[idx]
        sys_instruction = LANG_INSTRUCTIONS.get(lang, LANG_INSTRUCTIONS["en"])
        prompt = f"{sys_instruction}\nText: {self.texts[idx]}\nNeutral Rewrite:"
        return prompt, lang


class MT0Detoxifier:
    """
    A class for detoxifying text using the MT0 model with strict language-specific 
    prompts and lexical filtering.
    """

    LANGUAGES = Literal[
        "zh", "es", "ru", "ar", "hi", "uk", "de", "am",
        "en", "it", "ja", "he", "fr", "tt", "hin",
    ]

    def __init__(self, model_name: str = "s-nlp/mt0-xl-detox-orpo"):
        self.device = torch.device("cuda" if torch.cuda.is_available() 
                                   else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.lexicon = get_toxic_lexicon()

    def _prepare_batch(self, batch_texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of texts and prepare for model input."""
        return self.tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,  # Increased from 128 to accommodate longer instructions
        ).to(self.device)

    def _generate_batch(self, encodings: Dict[str, torch.Tensor]) -> List[str]:
        """Generate detoxified text for a batch of inputs."""
        with torch.no_grad():
            outputs = self.model.generate(
                **encodings,
                max_length=128,
                num_beams=5,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                num_beam_groups=5,
                diversity_penalty=2.5,
                num_return_sequences=1,
                early_stopping=True,
                trust_remote_code=True
            )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def apply_lexical_filter(self, text: str) -> str:
        """Removes known toxic words from the generated text."""
        for word in self.lexicon:
            # Replaces toxic words with empty string or space to nullify toxicity penalty
            text = re.sub(rf"\b{re.escape(word)}\b", "", text, flags=re.IGNORECASE)
        # Clean up double spaces that might be left behind
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def detoxify_batch(
        self, texts: List[str], langs: List[LANGUAGES], batch_size: int = 16
    ) -> List[str]:
        """Detoxify a batch of texts with their corresponding languages."""
        
        # Prepare prompts using strict instructions instead of basic prefix
        batch_texts =[
            f"{LANG_INSTRUCTIONS.get(lang, LANG_INSTRUCTIONS['en'])}\nText: {text}\nNeutral Rewrite:"
            for text, lang in zip(texts, langs)
        ]

        results =[]

        for i in tqdm(
            range(0, len(batch_texts), batch_size), desc="Processing batches"
        ):
            current_batch = batch_texts[i : i + batch_size]
            encodings = self._prepare_batch(current_batch)
            batch_results = self._generate_batch(encodings)
            
            # Apply post-processing lexical filter before adding to results
            clean_results =[self.apply_lexical_filter(res) for res in batch_results]
            results.extend(clean_results)

        return results


def process_file(input_path: str, output_path: str, batch_size: int = 16):
    """Process an input TSV file, detoxify the toxic sentences, and save to output file."""
    detoxifier = MT0Detoxifier()
    df = pd.read_csv(input_path, sep="\t")

    neutral_sentences = detoxifier.detoxify_batch(
        texts=df["toxic_sentence"].tolist(),
        langs=df["lang"].tolist(),
        batch_size=batch_size,
    )

    df["neutral_sentence"] = neutral_sentences
    df.to_csv(output_path, sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Detoxify text using upgraded MT0 baseline model"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=Path(INPUT_DATA_PATH, "dev_inputs.tsv"),
        help="Path to input TSV file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=Path(OUTPUT_DATA_PATH, "baseline_mt0_upgraded_dev.tsv"),
        help="Path to output TSV file",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16, # Adjusted default for safer memory with longer prompts
        help="Batch size for processing (default: 16)",
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    process_file(args.input_path, args.output_path, args.batch_size)
    print(f"Processing complete. Results saved to {output_path}")


if __name__ == "__main__":
    main()