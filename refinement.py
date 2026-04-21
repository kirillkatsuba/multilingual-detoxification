import argparse
import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoModelForSequenceClassification
)
from sentence_transformers import SentenceTransformer, util

# Same strict instructions as before
QWEN_INSTRUCTIONS = {
    "en": "Task: Detoxify the following English text. Rewrite it to be completely neutral, removing all profanity while strictly preserving the original meaning. Reply only in English.",
    "es": "Tarea: Desintoxicar el siguiente texto en español. Reescríbelo para que sea completamente neutral, eliminando todas las malas palabras mientras preservas estrictamente el significado original. Responde solo en español.",
    "ru": "Задача: Сделай следующий русский текст нетоксичным. Перепиши его так, чтобы он стал полностью нейтральным, удали всю нецензурную лексику, строго сохранив первоначальный смысл. Отвечай только на русском языке.",
    "uk": "Завдання: Зробіть наступний український текст нетоксичним. Перепишіть його так, щоб він став повністю нейтральним, видаліть усю нецензурну лексику, суворо зберігши початковий зміст. Відповідайте лише українською мовою.",
    "hin": "Task: Detoxify the following Hinglish text. Rewrite it to be completely neutral, removing all profanity while strictly preserving the original meaning. You MUST reply strictly in Hinglish (Roman script).",
    "am": "Task: Detoxify the following Amharic text. Rewrite it to be completely neutral, removing all profanity while strictly preserving the original meaning. You MUST reply strictly in Amharic script.",
    "tt": "Task: Detoxify the following Tatar text. Rewrite it to be completely neutral, removing all profanity while strictly preserving the original meaning. You MUST reply strictly in Tatar (Cyrillic script).",
    # Add other languages as needed...
}

def calculate_joint_score(toxic_text, neutral_candidate, labse_model, tox_model, tox_tokenizer, device):
    """Calculates the competition score for a single candidate."""
    # 1. Similarity
    embs = labse_model.encode([toxic_text, neutral_candidate], convert_to_tensor=True, device=device)
    sim = max(0.0, util.cos_sim(embs[0], embs[1]).item())
    
    # 2. Toxicity (Judge's model)
    tox_in = tox_tokenizer([neutral_candidate], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        tox_probs = torch.softmax(tox_model(**tox_in).logits, dim=-1)
        tox_score = tox_probs[0][1].item()
        
    return sim * (1.0 - tox_score)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mt0_results_path", type=str, required=True, help="Path to your mT0 output TSV")
    parser.add_argument("--output_path", type=str, default="final_refined_submission.tsv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(args.mt0_results_path, sep="\t")

    # --- STEP 1: Load Judge Models ---
    print("Loading Judging Models...")
    labse = SentenceTransformer("sentence-transformers/LaBSE").to(device)
    tox_tokenizer = AutoTokenizer.from_pretrained("textdetox/xlmr-large-toxicity-classifier-v2")
    tox_model = AutoModelForSequenceClassification.from_pretrained(
        "textdetox/xlmr-large-toxicity-classifier-v2", torch_dtype=torch.float16
    ).to(device)

    # --- STEP 2: Load Refiner Model (Qwen 3B) ---
    print("Loading Refiner Model (Qwen-3B)...")
    qwen_name = "Qwen/Qwen2.5-3B-Instruct"
    qwen_model = AutoModelForCausalLM.from_pretrained(qwen_name, torch_dtype=torch.float16).to(device)
    qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_name)

    refined_count = 0
    final_sentences = []

    # --- STEP 3: Selective Processing ---
    print("Starting Selective Refinement...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        toxic_text = row['toxic_sentence']
        lang = row['lang']
        current_neutral = str(row['neutral_sentence'])
        
        # Calculate current score
        current_score = calculate_joint_score(toxic_text, current_neutral, labse, tox_model, tox_tokenizer, device)
        
        # If score is good (e.g., > 0.7), keep it. If bad, try to refine.
        if current_score > 0.75:
            final_sentences.append(current_neutral)
        else:
            # REFINE with Qwen
            refined_count += 1
            sys_inst = QWEN_INSTRUCTIONS.get(lang, QWEN_INSTRUCTIONS["en"])
            prompt = qwen_tokenizer.apply_chat_template([
                {"role": "system", "content": sys_inst},
                {"role": "user", "content": f"Text: '{toxic_text}'\nNeutral Rewrite:"}
            ], tokenize=False, add_generation_prompt=True)
            
            inputs = qwen_tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = qwen_model.generate(**inputs, max_new_tokens=100, do_sample=False)
            
            qwen_neutral = qwen_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            
            # Final check: Is Qwen actually better?
            new_score = calculate_joint_score(toxic_text, qwen_neutral, labse, tox_model, tox_tokenizer, device)
            
            if new_score > current_score:
                final_sentences.append(qwen_neutral)
            else:
                final_sentences.append(current_neutral)

    # Save
    df['neutral_sentence'] = final_sentences
    df.to_csv(args.output_path, sep="\t", index=False)
    print(f"Refinement complete. Processed {len(df)} rows, attempted to refine {refined_count}. Results saved to {args.output_path}")

if __name__ == "__main__":
    main()