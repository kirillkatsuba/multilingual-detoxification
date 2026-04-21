import argparse
import gc
from pathlib import Path
from typing import List

import pandas as pd
import torch
import langid
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

# ---------------------------------------------------------
# 1. MT0 Short Prompts
# ---------------------------------------------------------
MT0_PROMPTS = {
    "zh": "排毒：", "es": "Desintoxicar: ", "ru": "Детоксифицируй: ",
    "ar": "إزالة السموم: ", "hi": "विषहरण: ", "uk": "Детоксифікуй: ",
    "de": "Entgiften: ", "am": "መርዝ መርዝ: ", "en": "Detoxify: ",
    "it": "Disintossicare: ", "ja": "解毒: ", "he": "לְסַלֵק רַעַל: ",
    "fr": "Désintoxiquer:", "tt": "Токсиннарны чыгару: ", "hin": "Detoxify: ",
}

# ---------------------------------------------------------
# 2. Qwen Strict Prompts
# ---------------------------------------------------------
QWEN_INSTRUCTIONS = {
    "en": "Task: Detoxify the following English text. Rewrite it to be completely neutral, removing all profanity while strictly preserving the original meaning. Reply only in English.",
    "es": "Tarea: Desintoxicar el siguiente texto en español. Reescríbelo para que sea completamente neutral, eliminando todas las malas palabras mientras preservas estrictamente el significado original. Responde solo en español.",
    "ru": "Задача: Сделай следующий русский текст нетоксичным. Перепиши его так, чтобы он стал полностью нейтральным, удали всю нецензурную лексику, строго сохранив первоначальный смысл. Отвечай только на русском языке.",
    "zh": "任务：将以下中文文本进行解毒。将其改写为完全中性的表达，删除所有脏话，同时严格保留原意。请只用中文回复。",
    "ja": "タスク：以下の日本語のテキストを無毒化してください。すべての冒とく的な表現を取り除き、元の意味を厳密に保ちながら、完全に中立的な表現に書き換えてください。日本語でのみ返信してください。",
    "ar": "مهمة: إزالة السموم من النص العربي التالي. أعد كتابته ليكون محايدًا تمامًا، وقم بإزالة جميع الألفاظ النابية مع الحفاظ بدقة على المعنى الأصلي. أجب باللغة العربية فقط.",
    "fr": "Tâche : Désintoxiquer le texte français suivant. Réécrivez-le pour qu'il soit complètement neutre, en supprimant tous les mots grossiers tout en préservant strictement le sens original. Répondez uniquement en français.",
    "it": "Compito: Disintossicare il seguente testo in italiano. Riscrivilo in modo che sia completamente neutrale, rimuovendo tutte le volgarità pur preservando rigorosamente il significato originale. Rispondi solo in italiano.",
    "de": "Aufgabe: Entgiften Sie den folgenden deutschen Text. Schreiben Sie ihn völlig neutral um, entfernen Sie alle Obszönitäten und behalten Sie dabei die ursprüngliche Bedeutung strikt bei. Antworten Sie nur auf Deutsch.",
    "uk": "Завдання: Зробіть наступний український текст нетоксичним. Перепишіть його так, щоб він став повністю нейтральним, видаліть усю нецензурну лексику, суворо зберігши початковий зміст. Відповідайте лише українською мовою.",
    "he": "משימה: נקה את הטקסט הבא בעברית מרעילות. שכתב אותו כך שיהיה ניטרלי לחלוטין, הסר את כל הקללות תוך שמירה קפדנית על המשמעות המקורית. השב בעברית בלבד.",
    "hi": "कार्य: निम्नलिखित हिंदी पाठ को गैर-विषाक्त बनाएं। सभी अपशब्दों को हटाते हुए, मूल अर्थ को सख्ती से बनाए रखते हुए, इसे पूरी तरह से तटस्थ बनाने के लिए फिर से लिखें। केवल हिंदी में उत्तर दें。",
    "hin": "Task: Detoxify the following Hinglish text. Rewrite it to be completely neutral, removing all profanity while strictly preserving the original meaning. You MUST reply strictly in Hinglish (Roman script).",
    "am": "Task: Detoxify the following Amharic text. Rewrite it to be completely neutral, removing all profanity while strictly preserving the original meaning. You MUST reply strictly in Amharic script.",
    "tt": "Task: Detoxify the following Tatar text. Rewrite it to be completely neutral, removing all profanity while strictly preserving the original meaning. You MUST reply strictly in Tatar (Cyrillic script)."
}

def free_memory():
    gc.collect()
    torch.cuda.empty_cache()

# ==============================================================================
# Model 1: mT0 Generator
# ==============================================================================
def run_mt0(df: pd.DataFrame, batch_size: int = 16) -> List[str]:
    print("\n--- [Pass 1/3] Running mT0-XL ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained("s-nlp/mt0-xl-detox-orpo", torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained("s-nlp/mt0-xl-detox-orpo")
    
    results =[]
    texts = df["toxic_sentence"].tolist()
    langs = df["lang"].tolist()
    
    prompts =[MT0_PROMPTS.get(lang, "Detoxify: ") + text for text, lang in zip(texts, langs)]
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="mT0 Inference"):
        batch = prompts[i: i + batch_size]
        encodings = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model.generate(**encodings, max_length=128, num_beams=5)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(decoded)
        
    del model, tokenizer
    free_memory()
    return results

# ==============================================================================
# Model 2: Qwen Generator
# ==============================================================================
def run_qwen(df: pd.DataFrame, batch_size: int = 8) -> List[str]:
    print("\n--- [Pass 2/3] Running Qwen2.5-7B-Instruct ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    
    results = []
    texts = df["toxic_sentence"].tolist()
    langs = df["lang"].tolist()
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Qwen Inference"):
        batch_texts = texts[i: i + batch_size]
        batch_langs = langs[i: i + batch_size]
        
        prompts =[]
        for text, lang in zip(batch_texts, batch_langs):
            sys_prompt = QWEN_INSTRUCTIONS.get(lang, QWEN_INSTRUCTIONS["en"])
            messages =[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Text: '{text}'\nNeutral Rewrite:"}
            ]
            prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False, temperature=0.0)
            
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[:, input_length:]
        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        results.extend([t.strip() for t in decoded])
        
    del model, tokenizer
    free_memory()
    return results

# ==============================================================================
# The Ranker (Selects Best Output using langid)
# ==============================================================================
def run_ranker(df: pd.DataFrame) -> List[str]:
    print("\n--- [Pass 3/3] Ranking Candidates ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Semantic Similarity Model
    print("Loading LaBSE for Meaning Preservation checks...")
    labse = SentenceTransformer("sentence-transformers/LaBSE").to(device)
    
    best_sentences =[]
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Ranking"):
        toxic = row['toxic_sentence']
        lang = row['lang']
        cand_mt0 = str(row['mt0_candidate']).strip()
        cand_qwen = str(row['qwen_candidate']).strip()
        
        # 1. Embed sentences
        embeddings = labse.encode([toxic, cand_mt0, cand_qwen], convert_to_tensor=True, device=device)
        sim_mt0 = util.cos_sim(embeddings[0], embeddings[1]).item()
        sim_qwen = util.cos_sim(embeddings[0], embeddings[2]).item()
        
        # 2. Check for Language Drift using langid
        # We only care if Qwen accidentally translated a non-English sentence into English
        if lang != "en":
            detected_lang, _ = langid.classify(cand_qwen)
            if detected_lang == "en":
                sim_qwen -= 0.5  # Massive penalty for translating to English
        
        # 3. Select Winner
        if sim_qwen > sim_mt0:
            best_sentences.append(cand_qwen)
        else:
            best_sentences.append(cand_mt0)
            
    del labse
    free_memory()
    return best_sentences

# ==============================================================================
# Main Pipeline execution
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="input_data/dev_inputs.tsv")
    parser.add_argument("--output_path", type=str, default="output_data/sol_ensemble_dev.tsv")
    args = parser.parse_args()

    df = pd.read_csv(args.input_path, sep="\t")
    
    # Pass 1: Generate mT0 Candidates
    df["mt0_candidate"] = run_mt0(df, batch_size=32)
    
    # Pass 2: Generate Qwen Candidates
    df["qwen_candidate"] = run_qwen(df, batch_size=8)
    
    # Pass 3: Rank and Pick Best
    df["neutral_sentence"] = run_ranker(df)
    
    # Clean up dataframe to match submission format (drop intermediate columns)
    final_df = df.drop(columns=["mt0_candidate", "qwen_candidate"])
    
    # Save Final Result
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.output_path, sep="\t", index=False)
    
    print(f"\n✅ Pipeline Complete! Best sentences saved to {args.output_path}")

if __name__ == "__main__":
    main()