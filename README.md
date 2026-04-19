# Multilingual Text Detoxification (TextDetox)

## Project Overview

**Task Type:** Multilingual Sequence-to-Sequence Task

**Description:** Given a toxic piece of text, re-write it in a non-toxic way while saving the main content as much as possible. The goal is to transform explicit toxic content into neutral, non-offensive paraphrases that preserve the meaningful content.

---

## Task Definition

### Problem Statement
Social networks like Facebook and Instagram try to address toxicity, but they usually simply block such texts. This project proposes a proactive approach: presenting a neutral version of a user message which preserves meaningful content. This task is denoted as **text detoxification**.

### Example Transformations

| Toxic Input | Detoxified Output |
|-------------|-------------------|
| he had steel b*lls too! | he was brave too! |
| delete the page and sh*t up | delete the page |
| what a chicken cr*p excuse for a reason. | what a bad excuse for a reason. |

### Definition of Toxicity
- **Explicit toxicity:** Obvious presence of obscene and rude lexicon where meaningful neutral content is still present
- **NOT handled:** Implicit toxicity (sarcasm, passive aggressiveness, direct hate where no neutral content can be found)

---

## Languages Supported (15 Total)

### Languages with Parallel Training Data (9)
- English (en)
- Spanish (es)
- German (de)
- Chinese (zh)
- Arabic (ar)
- Hindi (hi)
- Ukrainian (uk)
- Russian (ru)
- Amharic (am)

### New Languages WITHOUT Parallel Training Data (6)
- Italian (it)
- French (fr)
- Hebrew (he)
- Japanese (ja)
- Tatar (tt)
- Hinglish (hin) - code-switch language

**Challenge:** The main difficulty lies in creating detoxification systems for languages without parallel training data (cross-lingual transfer).

---

## Data Resources

### Training Data

| Dataset | Description | Link |
|---------|-------------|------|
| **Parallel Text Detoxification Dataset** | 400 toxic-neutral pairs for each of 9 languages | [HuggingFace](https://huggingface.co/datasets/textdetox/multilingual_paradetox) |
| **Toxicity Classification Dataset** | Toxicity classification data for all 15 languages | [HuggingFace](https://huggingface.co/datasets/textdetox/multilingual_toxicity_dataset) |
| **Toxic Keywords List** | Toxic keywords for all 15 languages | [HuggingFace](https://huggingface.co/datasets/textdetox/multilingual_toxic_lexicon) |
| **Toxic Spans** | Toxic spans for 9 languages | [HuggingFace](https://huggingface.co/datasets/textdetox/multilingual_toxic_spans) |

### Test Data
- Available at: [HuggingFace Test Set](https://huggingface.co/datasets/textdetox/multilingual_paradetox_test)
- 600 toxic sentences per language (for languages with training data)
- 100 toxic sentences per new language (Italian, French, Hebrew, Hinglish, Japanese, Tatar)

---

## Evaluation Metrics

The evaluation is based on three key parameters (each in range [0, 1]):

### 1. Style Transfer Accuracy (STA)
- Measures the **non-toxicity** of generated paraphrase
- Uses fine-tuned [xlm-roberta-large toxicity classifier](https://huggingface.co/textdetox/xlmr-large-toxicity-classifier-v2)

### 2. Content Preservation (SIM)
- Evaluates similarity of content between original and detoxified text
- Calculated as cosine similarity between [LaBSE](https://huggingface.co/sentence-transformers/LaBSE) embeddings

### 3. Fluency (FL)
- Estimates text adequacy and similarity to human-written detoxified references
- Uses [xCOMET](https://huggingface.co/myyycroft/XCOMET-lite) model (machine translation metric)

### Joint Score (J)
```
J = mean(STA * SIM * FL) per sample
```

The Joint score combines all three metrics for leaderboard ranking.

**2025 Update:** Metrics now incorporate human references for comparison, acknowledging that even human detoxifications may not be perfectly non-toxic.

---

## Baseline Methods

### Supervised Baselines

#### 1. Fine-tuned Seq2Seq Models
- **mT0-XL-detox:** [s-nlp/mt0-xl-detox-orpo](https://huggingface.co/s-nlp/mt0-xl-detox-orpo) - One of the top solutions from TextDetox2024
- **mBART Baseline:** [textdetox/mbart-detox-baseline](https://huggingface.co/textdetox/mbart-detox-baseline) - Fine-tuned on 9 languages
- **Fine-tuning Example:** [Google Colab Notebook](https://colab.research.google.com/drive/1Wd_32qGpED5M3cfmDapKqOGplgLQ39xP?usp=sharing)

#### 2. Reference Implementation
- ruT5 model for detoxification: [GitHub Example](https://github.com/s-nlp/russe_detox_2022/tree/main/baselines/t5)

### Unsupervised Baselines

| Method | Description |
|--------|-------------|
| **Duplicate** | Simple duplication of the toxic input (baseline lower bound) |
| **Delete** | Elimination of toxic keywords using predefined [dictionary](https://huggingface.co/datasets/textdetox/multilingual_toxic_lexicon) |
| **Backtranslation** | Translate → Detoxify in English → Translate back. Uses NLLB-600M for translation and [bart-base-detox](https://huggingface.co/s-nlp/bart-base-detox) for English detoxification |
| **LLM Prompting** | Zero-shot or few-shot prompting with LLMs (baseline: LLaMa-70B, GPT-4, GPT-4o, o3-mini) |

### Code Repository
All baseline code: [GitHub - PAN CLEF25 Text Detoxification](https://github.com/pan-webis-de/pan-code/tree/master/clef25/text-detoxification)

---

## How to Start the Project

### Step 1: Understand the Task
1. Read the task overview: [PAN CLEF 2025 Text Detoxification](https://pan.webis.de/clef25/pan25-web/text-detoxification.html)
2. Review the papers:
   - [TextDetox CLEF 2025 Overview Paper](https://ceur-ws.org/Vol-4038/paper_305.pdf)
   - [CEUR-WS Volume 4038](https://ceur-ws.org/Vol-4038/) (search for "Detox" papers)

### Step 2: Access the Data
```python
# Example: Load the parallel detoxification dataset
from datasets import load_dataset

# Training data
dataset = load_dataset("textdetox/multilingual_paradetox")

# Test data
test_dataset = load_dataset("textdetox/multilingual_paradetox_test")

# Toxic lexicon
lexicon = load_dataset("textdetox/multilingual_toxic_lexicon")
```

### Step 3: Explore Baseline Models
```python
# Example: Use the mT0-XL detox model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "s-nlp/mt0-xl-detox-orpo"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Example: Use mBART baseline
model_name = "textdetox/mbart-detox-baseline"
```

### Step 4: Run Evaluation
```bash
# Clone the evaluation repository
git clone https://github.com/pan-webis-de/pan-code.git
cd pan-code/clef25/text-detoxification

# Follow instructions in the README for evaluation scripts
```

### Step 5: Develop Your Solution

#### Recommended Approaches:

1. **For Languages with Training Data:**
   - Fine-tune multilingual seq2seq models (mT5, mBART, NLLB)
   - Use prompt-tuning or LoRA for efficient fine-tuning
   - Consider ensemble methods

2. **For Languages WITHOUT Training Data (Cross-lingual Transfer):**
   - Zero-shot cross-lingual transfer from multilingual models
   - Backtranslation approach
   - Few-shot prompting with multilingual LLMs
   - Translate-train-translate approach

3. **Hybrid Methods:**
   - Combine rule-based toxic word deletion with neural detoxification
   - Use toxic lexicon to guide the detoxification process

---

## Submission Requirements

### File Format
- **Format:** .zip file containing a .tsv file
- **Columns:** `toxic_text`, `neutral_text`, `lang`
- **Requirements:**
  - Fill ALL rows (no NaN values)
  - Do not edit `neutral_text` and `lang` columns from original input
  - Do not remove lines (size must match original file)

### Leaderboard Categories
- **AvgP:** Average score on languages with parallel data (9 languages)
- **AvgNP:** Average score on new languages without parallel data (6 languages)

### Submission Platform
- [CodaLab Competition](https://codalab.lisn.upsaclay.fr/competitions/22396)
- Indicate team as: **"Skoltech--26"**

---

## Key Resources Summary

| Resource | Link |
|----------|------|
| **Task Page** | https://pan.webis.de/clef25/pan25-web/text-detoxification.html |
| **Training Data** | https://huggingface.co/datasets/textdetox/multilingual_paradetox |
| **Test Data** | https://huggingface.co/datasets/textdetox/multilingual_paradetox_test |
| **Toxic Lexicon** | https://huggingface.co/datasets/textdetox/multilingual_toxic_lexicon |
| **Baseline Code** | https://github.com/pan-webis-de/pan-code/tree/master/clef25/text-detoxification |
| **mT0-XL Model** | https://huggingface.co/s-nlp/mt0-xl-detox-orpo |
| **mBART Model** | https://huggingface.co/textdetox/mbart-detox-baseline |
| **Toxicity Classifier** | https://huggingface.co/textdetox/xlmr-large-toxicity-classifier-v2 |
| **Colab Template** | https://colab.research.google.com/drive/1ttPT6X4K0ovgbzmNjlcEiprkj1LaBuF2 |
| **CodaLab Submission** | https://codalab.lisn.upsaclay.fr/competitions/22396 |
| **HuggingFace Space** | https://huggingface.co/textdetox |
| **Google Group** | https://groups.google.com/g/textdetox-clef2025 |

---

## Recommended Papers to Read

1. [Overview of PAN 2025: Generative AI Detection, Multilingual Text Detoxification](https://dl.acm.org/doi/10.1007/978-3-031-88720-8_64)
2. [TextDetox CLEF 2025 Paper](https://ceur-ws.org/Vol-4038/paper_305.pdf)
3. [Overview of Multilingual Text Detoxification Task at PAN 2025](https://nchr.elsevierpure.com/en/publications/overview-of-the-multilingual-text-detoxification-task-at-pan-2025)
4. [CEUR-WS Volume 4038](https://ceur-ws.org/Vol-4038/) - All TextDetox papers

---

## Environment Setup

### Google Colab (Recommended)
The assignment template is designed for Google Colab:
- [Colab Template](https://colab.research.google.com/drive/1ttPT6X4K0ovgbzmNjlcEiprkj1LaBuF2)

### Local Setup
```bash
# Create virtual environment
python -m venv textdetox
source textdetox/bin/activate

# Install dependencies
pip install transformers datasets torch sentencepiece
pip install sentence-transformers  # For LaBSE
pip install comet  # For xCOMET evaluation

# Clone evaluation code
git clone https://github.com/pan-webis-de/pan-code.git
```

### GPU Requirements
- Standard Google Colab (free tier with T4 GPU) should work
- Consumer-grade GPU like NVIDIA 3090 RTX is sufficient
- For larger models (mT0-XL), may need A100 or multi-GPU setup

---

## Project Timeline (From Assignment)

| Date | Milestone |
|------|-----------|
| April 25, 2025 | Registration closes |
| May 8, 2025 | Test phase starts |
| May 23, 2025 | Final submissions deadline |
| May 30, 2025 | Participants paper submission |
| June 27, 2025 | Notification of acceptance |
| July 7, 2025 | Camera-ready due |
| September 9-12, 2025 | CLEF Conference in Madrid, Spain |

---

## Tips for Success

1. **Start Simple:** Begin with the provided baseline models (mT0-XL or mBART)
2. **Focus on Cross-lingual Transfer:** The main challenge is languages without training data
3. **Use All Available Resources:** Combine toxic lexicon, classification datasets, and parallel data
4. **Balance All Metrics:** Optimize for Joint score, not just individual metrics
5. **Experiment with Prompting:** LLM prompting can be effective, especially for new languages
6. **Check Output Quality:** Ensure outputs are fluent and preserve content meaning

---

## Contact & Support

- **Google Group:** https://groups.google.com/g/textdetox-clef2025
- **HuggingFace Space:** https://huggingface.co/textdetox
- **Telegram Group:** Available through Canvas (for course participants)

---