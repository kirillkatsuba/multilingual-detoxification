import argparse
import gc
import math
import concurrent.futures
from pathlib import Path
from typing import List

import pandas as pd
import torch
import torch.multiprocessing as mp
import langid
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig
)
from sentence_transformers import SentenceTransformer, util

# ---------------------------------------------------------
# Prompts & Instructions
# ---------------------------------------------------------
MT0_PROMPTS = {
    "zh": "排毒：", "es": "Desintoxicar: ", "ru": "Детоксифицируй: ",
    "ar": "إزالة السموم: ", "hi": "विषहरण: ", "uk": "Детоксифікуй: ",
    "de": "Entgiften: ", "am": "መርዝ መርዝ: ", "en": "Detoxify: ",
    "it": "Disintossicare: ", "ja": "解毒: ", "he": "לְסַלֵק רַעַל: ",
    "fr": "Désintoxiquer:", "tt": "Токсиннарны чыгару: ", "hin": "Detoxify: ",
}

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
# Model 1: mT0 Generator (Targeted to Specific GPU)
# ==============================================================================
def run_mt0(df: pd.DataFrame, device_id: int, batch_size: int = 16) -> List[str]:
    device = torch.device(f"cuda:{device_id}")
    # float16 is optimal for Kaggle T4 GPUs
    model = AutoModelForSeq2SeqLM.from_pretrained("s-nlp/mt0-xl-detox-orpo", torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained("s-nlp/mt0-xl-detox-orpo")
    
    results = []
    texts = df["toxic_sentence"].tolist()
    langs = df["lang"].tolist()
    prompts =[MT0_PROMPTS.get(lang, "Detoxify: ") + text for text, lang in zip(texts, langs)]
    
    for i in tqdm(range(0, len(prompts), batch_size), position=device_id, desc=f"GPU {device_id} [mT0]"):
        batch = prompts[i: i + batch_size]
        encodings = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model.generate(**encodings, max_length=128, num_beams=3) # Reduced beams slightly for speed
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(decoded)
        
    del model, tokenizer
    free_memory()
    return results

# ==============================================================================
# Model 2: Qwen Generator (4-bit to fit Kaggle T4)
# ==============================================================================
def run_qwen(df: pd.DataFrame, device_id: int, batch_size: int = 8) -> List[str]:
    device = torch.device(f"cuda:{device_id}")
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    # Crucial for Kaggle T4 GPUs: Load in 4-bit!
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        quantization_config=quant_config, 
        device_map={"": device_id}  # Force to specific GPU
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    
    results = []
    texts = df["toxic_sentence"].tolist()
    langs = df["lang"].tolist()
    
    for i in tqdm(range(0, len(texts), batch_size), position=device_id, desc=f"GPU {device_id}[Qwen]"):
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
# The Ranker 
# ==============================================================================
def run_ranker(df: pd.DataFrame, device_id: int) -> List[str]:
    device = torch.device(f"cuda:{device_id}")
    labse = SentenceTransformer("sentence-transformers/LaBSE").to(device)
    best_sentences =[]
    
    for _, row in tqdm(df.iterrows(), total=len(df), position=device_id, desc=f"GPU {device_id} [Rank]"):
        toxic = row['toxic_sentence']
        lang = row['lang']
        cand_mt0 = str(row['mt0_candidate']).strip()
        cand_qwen = str(row['qwen_candidate']).strip()
        
        # 1. Embed sentences
        embeddings = labse.encode([toxic, cand_mt0, cand_qwen], convert_to_tensor=True, device=device)
        sim_mt0 = util.cos_sim(embeddings[0], embeddings[1]).item()
        sim_qwen = util.cos_sim(embeddings[0], embeddings[2]).item()
        
        # 2. Check for Language Drift (langid)
        if lang != "en":
            detected_lang, _ = langid.classify(cand_qwen)
            if detected_lang == "en":
                sim_qwen -= 0.5  # Penalty for translating to English
        
        # 3. Select Winner
        if sim_qwen > sim_mt0:
            best_sentences.append(cand_qwen)
        else:
            best_sentences.append(cand_mt0)
            
    del labse
    free_memory()
    return best_sentences

# ==============================================================================
# Worker Function (Executes entire pipeline for one chunk of data)
# ==============================================================================
def process_chunk(df_chunk: pd.DataFrame, device_id: int) -> pd.DataFrame:
    df_chunk = df_chunk.copy()
    print(f"\n[GPU {device_id}] Starting mT0 Pass...")
    df_chunk["mt0_candidate"] = run_mt0(df_chunk, device_id, batch_size=16)
    
    print(f"\n[GPU {device_id}] Starting Qwen Pass...")
    df_chunk["qwen_candidate"] = run_qwen(df_chunk, device_id, batch_size=8)
    
    print(f"\n[GPU {device_id}] Starting Ranking Pass...")
    df_chunk["neutral_sentence"] = run_ranker(df_chunk, device_id)
    
    # Drop intermediate columns
    return df_chunk.drop(columns=["mt0_candidate", "qwen_candidate"])

# ==============================================================================
# Main Pipeline 
# ==============================================================================
def main():
    # VERY IMPORTANT: Sets multiprocessing context for PyTorch CUDA handling
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="input_data/dev_inputs.tsv")
    parser.add_argument("--output_path", type=str, default="output_data/sol_ensemble_multi_dev.tsv")
    args = parser.parse_args()

    df = pd.read_csv(args.input_path, sep="\t")
    
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print(f"Warning: Only {num_gpus} GPU(s) found. Script will run on a single GPU.")
        num_gpus = 1

    print(f"Total rows: {len(df)}. Splitting across {num_gpus} GPU(s)...")

    # Split dataframe evenly across GPUs
    chunk_size = math.ceil(len(df) / num_gpus)
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    results =[]
    
    # Run the worker processes in parallel
    ctx = mp.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_gpus, mp_context=ctx) as executor:
        futures =[
            executor.submit(process_chunk, chunk, gpu_id) 
            for gpu_id, chunk in enumerate(chunks)
        ]
        
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            
    # Recombine datasets, sort by original index
    final_df = pd.concat(results).sort_index()
    
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.output_path, sep="\t", index=False)
    print(f"\n✅ Multi-GPU Pipeline Complete! Best sentences saved to {args.output_path}")

if __name__ == "__main__":
    main()