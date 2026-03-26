# SmartGuard — Full Coding Agent Specification
# Track B: Train Your Own Classifier

---

## DOCUMENT INDEX

- [S0 — Project Overview](#s0)
- [S1 — Dataset Facts (read first)](#s1)
- [S2 — Repo Structure](#s2)
- [S3 — Dependencies & Environment](#s3)
- [S4 — Phase 1: Dataset Preparation (`prepare_data.py`)](#s4)
- [S5 — Phase 2: Stratified Split (inside `prepare_data.py`)](#s5)
- [S6 — Phase 3–4: Encoding + Tokenization (`train.py`)](#s6)
- [S7 — Phase 5: Model Training (`train.py`)](#s7)
- [S8 — Phase 6: Evaluation (`eval.py`)](#s8)
- [S9 — Phase 7: Error Analysis (`eval.py`)](#s9)
- [S10 — Phase 8–9: Latency + Baseline Comparison (`eval.py`)](#s10)
- [S11 — Red-Team Suite (`red_team_suite.csv`)](#s11)
- [S12 — Phase 10: Streamlit Dashboard (`app.py`)](#s12)
- [S13 — Supporting Files](#s13)
- [S14 — Research Questions Mapping](#s14)
- [S15 — PPT Slide Mapping](#s15)
- [S16 — Final Checklist](#s16)

---

## S0 — Project Overview

**Project name:** SmartGuard  
**Goal:** Build an LLM input/output firewall that classifies any prompt as safe or harmful, runs on CPU only, and includes a structured red-team evaluation.  
**Track:** Track B — train your own classifier from scratch via fine-tuning  
**Base model:** distilbert-base-uncased (66M params, 6 transformer layers)  
**Task:** 5-class sequence classification  
**Classes:** safe, jailbreak, injection, toxic, pii  
**Random seed (pin everywhere):** 42

---

## S1 — Dataset Facts (read first before writing any code)

### S1.1 — Source files

| File | Label | Rows | Avg text len | Source dataset |
|---|---|---|---|---|
| `data/raw/jailbreak.csv` | jailbreak | 100 | 76 chars | JBB-Behaviors |
| `data/raw/indirect.csv` | injection | 100 | 104 chars | BIPIA |
| `data/raw/PII.csv` | pii | 100 | 63 chars | Nemotron-PII |
| `data/raw/toxic.csv` | toxic | 100 | 54 chars | ToxiGen + Civil-Comments |
| `data/raw/safe.csv` | safe | 100 | 43 chars | Alpaca-Cleaned |

**Total: 500 rows. Dataset is perfectly balanced at 100 rows per class.**

### S1.2 — CSV parsing quirks (CRITICAL — plain pd.read_csv will return all NaN)

Every file has the following structure:
- Lines 1–7: completely blank (`,,,,,\r\n` repeated 7 times)
- Line 8: the real header row
- Lines 9+: data rows with trailing backslash (`\\\r\n`) instead of normal line endings

Additional quirk in `PII.csv` only:
- The header line is prefixed with an RTF artifact: `\f0\fs24 \cf0 id,text,...`
- Strip everything before `id,` using a regex before parsing

### S1.3 — Column schema (after parsing)

All five files have identical columns: `id, text, label, pattern, source, split`

- **Keep:** `text`, `label`
- **Drop:** `id`, `pattern`, `source`, `split`

The `split` column in the source files marks all rows as `"train"` — this is from the original data source. Ignore it. Do your own 70/15/15 split from scratch.

### S1.4 — Label values (exact strings, lowercase)

- `safe`
- `jailbreak`
- `injection`
- `toxic`
- `pii`

### S1.5 — Important data characteristics for research notes

- Safe texts are the shortest class (avg 43 chars) — potential length bias confound
- Injection texts are the longest (avg 104 chars)
- toxic and jailbreak classes have semantic overlap — expect confusion between them
- injection and jailbreak classes also overlap (both involve adversarial override attempts)
- With 350 training rows and 66M params, overfitting is expected — document it

---

## S2 — Repo Structure

Create exactly this directory and file layout:

```
smartguard/
├── data/
│   ├── raw/
│   │   ├── jailbreak.csv
│   │   ├── indirect.csv
│   │   ├── PII.csv
│   │   ├── toxic.csv
│   │   └── safe.csv
│   └── processed/
│       ├── final_dataset.csv
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       └── dataset_stats.json
├── model/
│   ├── config.json
│   ├── pytorch_model.bin  (or model.safetensors)
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   └── special_tokens_map.json
├── results/
│   ├── eval_results.json
│   ├── training_log.csv
│   ├── red_team_results.csv
│   └── confusion_matrix.png
├── red_team_suite.csv
├── label2id.json
├── prepare_data.py
├── train.py
├── eval.py
├── app.py
├── requirements.txt
└── README.md
```

---

## S3 — Dependencies & Environment

### S3.1 — requirements.txt (pin exact versions)

```
# random_state=42 used throughout — do not change
torch==2.3.0
transformers==4.41.2
datasets==2.20.0
scikit-learn==1.5.0
pandas==2.2.2
numpy==1.26.4
streamlit==1.35.0
plotly==5.22.0
matplotlib==3.9.0
seaborn==0.13.2
```

### S3.2 — Python version

Python 3.10 or higher required.

### S3.3 — Install command

```bash
pip install -r requirements.txt
```

### S3.4 — Hardware constraint

All inference must run on CPU only. Do NOT use `.cuda()` or `.to("gpu")` anywhere. Use `device = torch.device("cpu")` explicitly.

---

## S4 — Phase 1: Dataset Preparation

**File:** `prepare_data.py`  
**Run:** `python prepare_data.py`  
**Outputs:** `data/processed/final_dataset.csv`, `data/processed/dataset_stats.json`

### S4.1 — Parsing function (apply to all 5 files)

```python
import pandas as pd
import re
from io import StringIO

def parse_smartguard_csv(path: str) -> pd.DataFrame:
    """
    Handles the 7-blank-line header, trailing backslashes,
    and RTF artifact in PII.csv.
    """
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    # Find the real header: first line containing both 'text' and 'label'
    header_idx = None
    for i, line in enumerate(lines):
        if 'text' in line and 'label' in line:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"Could not find header in {path}")

    # Slice from header onward
    data_lines = lines[header_idx:]

    # Strip trailing backslashes and normalize line endings
    data_lines = [line.rstrip('\\\r\n') + '\n' for line in data_lines]

    # Strip RTF prefix from header line (present in PII.csv)
    data_lines[0] = re.sub(r'^.*?id,', 'id,', data_lines[0])

    content = ''.join(data_lines)
    df = pd.read_csv(StringIO(content))
    return df
```

### S4.2 — Load and label each file

```python
FILE_MAP = {
    'jailbreak': 'data/raw/jailbreak.csv',
    'injection': 'data/raw/indirect.csv',
    'pii':       'data/raw/PII.csv',
    'toxic':     'data/raw/toxic.csv',
    'safe':      'data/raw/safe.csv',
}
```

For each file: parse with `parse_smartguard_csv`, keep only `text` and `label` columns, verify the label column matches the expected label for that file.

### S4.3 — Merge all five into one DataFrame

Concatenate all five DataFrames with `pd.concat(..., ignore_index=True)`. Result must have exactly 500 rows and two columns: `text`, `label`.

### S4.4 — Validate labels

After merging, assert that `df['label'].unique()` contains exactly and only: `{'safe', 'jailbreak', 'injection', 'toxic', 'pii'}`.

Print value counts. Expected output:
```
safe          100
jailbreak     100
injection     100
toxic         100
pii           100
```

### S4.5 — Clean the data

Apply all of the following in order:

1. Drop rows where `text` is null or NaN
2. Strip leading/trailing whitespace from `text`: `df['text'] = df['text'].str.strip()`
3. Collapse internal whitespace (tabs, multiple spaces, newlines): `df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)`
4. Drop rows where `text` is empty string after stripping
5. Drop rows where `len(text) < 10`
6. Drop duplicate rows based on `text` column only (keep first)
7. Reset index: `df.reset_index(drop=True)`

Print before and after row count. Document the delta.

### S4.6 — Exploratory stats (save to dataset_stats.json)

Compute and save:
- Total rows after cleaning
- Rows per class (dict)
- Mean text length per class (dict)
- Min text length per class (dict)
- Max text length per class (dict)

This data is used in PPT Slide 9 and the README dataset section.

### S4.7 — Save final_dataset.csv

Save with exactly two columns: `text`, `label`. No index column.

```python
df.to_csv('data/processed/final_dataset.csv', index=False)
```

---

## S5 — Phase 2: Stratified Split

**Runs inside:** `prepare_data.py` (after S4)  
**Outputs:** `data/processed/train.csv`, `data/processed/val.csv`, `data/processed/test.csv`

### S5.1 — Split ratios

- Train: 70% (~350 rows)
- Validation: 15% (~75 rows)
- Test: 15% (~75 rows)

### S5.2 — Two-step stratified split procedure

Use `sklearn.model_selection.train_test_split` with `stratify=y` in both steps.

```python
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

X = df['text']
y = df['label']

# Step 1: split off 15% test
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=RANDOM_SEED
)

# Step 2: split remaining 85% into train (82.4%) and val (17.6%)
# 0.176 * 0.85 ≈ 0.15 of total
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.176, stratify=y_trainval, random_state=RANDOM_SEED
)
```

### S5.3 — Save splits

```python
pd.DataFrame({'text': X_train, 'label': y_train}).to_csv('data/processed/train.csv', index=False)
pd.DataFrame({'text': X_val,   'label': y_val  }).to_csv('data/processed/val.csv',   index=False)
pd.DataFrame({'text': X_test,  'label': y_test }).to_csv('data/processed/test.csv',  index=False)
```

### S5.4 — Verify splits

After saving, print class distribution for all three splits. Expected: approximately 70/15/15 per class in every split.

### S5.5 — Update dataset_stats.json

Append split sizes to the JSON file created in S4.6:
```json
{
  "train_size": 350,
  "val_size": 75,
  "test_size": 75,
  "train_per_class": { "safe": 70, ... },
  "val_per_class": { "safe": 15, ... },
  "test_per_class": { "safe": 15, ... }
}
```

### S5.6 — Lock test set

Add a comment in the code directly above where test.csv is saved:
```python
# TEST SET LOCKED — do not use for any decisions until final evaluation in eval.py
```

---

## S6 — Phase 3–4: Label Encoding + Tokenization

**File:** `train.py`  
**Runs before training**

### S6.1 — Label encoding

```python
LABEL2ID = {
    "safe":      0,
    "jailbreak": 1,
    "injection": 2,
    "toxic":     3,
    "pii":       4,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
```

Save to `label2id.json`:
```python
import json
with open('label2id.json', 'w') as f:
    json.dump({"label2id": LABEL2ID, "id2label": ID2LABEL}, f, indent=2)
```

### S6.2 — Load tokenizer

```python
from transformers import AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
```

### S6.3 — Tokenization settings

- `max_length = 128`  (rationale: injection texts avg 104 chars ≈ ~25 tokens; 128 is safe headroom)
- `truncation = True`
- `padding = "max_length"`

### S6.4 — Tokenize function

```python
def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )
```

### S6.5 — Build HuggingFace Dataset objects

Load each CSV, map labels to integers, convert to HuggingFace Dataset, apply tokenize function.

```python
from datasets import Dataset

def load_split(path):
    df = pd.read_csv(path)
    df['labels'] = df['label'].map(LABEL2ID)
    ds = Dataset.from_pandas(df[['text', 'labels']])
    return ds.map(tokenize, batched=True)

train_ds = load_split('data/processed/train.csv')
val_ds   = load_split('data/processed/val.csv')
test_ds  = load_split('data/processed/test.csv')
```

### S6.6 — Verification step (do not skip)

Before training, print one example from each class decoded back to text. Confirm labels are integers and text decodes correctly. This catches encoding bugs silently.

---

## S7 — Phase 5: Model Training

**File:** `train.py`  
**Run:** `python train.py`  
**Outputs:** `model/` directory with weights, `results/training_log.csv`

### S7.1 — Load model

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=5,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)
```

### S7.2 — Training arguments

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./model",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_steps=50,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    seed=42,
    no_cuda=True,           # CPU only — mandatory
    report_to="none",       # disable wandb/tensorboard
)
```

### S7.3 — Compute metrics function

```python
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "eval_f1": f1_score(labels, preds, average="macro"),
    }
```

### S7.4 — Early stopping callback

```python
from transformers import EarlyStoppingCallback

callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
```

### S7.5 — Trainer

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    callbacks=callbacks,
)
```

### S7.6 — Train and time it

```python
import time

start = time.time()
trainer.train()
training_time_seconds = time.time() - start
print(f"Training time: {training_time_seconds:.1f}s")
```

### S7.7 — Save model and tokenizer

```python
trainer.save_model("./model")
tokenizer.save_pretrained("./model")
```

### S7.8 — Save training log to CSV

Extract per-epoch training loss and validation loss from `trainer.state.log_history` and save to `results/training_log.csv`.

Expected columns: `epoch, train_loss, eval_loss, eval_accuracy, eval_f1`

This data is required for the loss curve plot (PPT Slide 11, RQ7).

### S7.9 — Print final val metrics

After training, print the best checkpoint's validation accuracy and macro F1. State which epoch achieved best val F1.

---

## S8 — Phase 6: Evaluation

**File:** `eval.py`  
**Run:** `python eval.py`  
**Outputs:** `results/eval_results.json`, `results/confusion_matrix.png`

### S8.1 — Load saved model for inference

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained("./model")
tokenizer = AutoTokenizer.from_pretrained("./model")
model.eval()
device = torch.device("cpu")
model.to(device)
```

### S8.2 — Inference function

```python
def predict(texts: list[str], threshold: float = 0.5):
    """
    Returns: list of (predicted_label, confidence_score, all_probs_dict)
    confidence_score = softmax probability of the predicted class
    If max prob < threshold, optionally flag as uncertain
    """
    inputs = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).cpu().numpy()

    results = []
    for p in probs:
        pred_id = p.argmax()
        label = ID2LABEL[pred_id]
        confidence = float(p[pred_id])
        all_probs = {ID2LABEL[i]: float(p[i]) for i in range(5)}
        results.append((label, confidence, all_probs))
    return results
```

### S8.3 — Run evaluation on test set

Load `data/processed/test.csv`. Run `predict()` on all texts. Compute:

- `sklearn.metrics.classification_report(y_true, y_pred, output_dict=True)` → per-class precision, recall, F1
- `sklearn.metrics.confusion_matrix(y_true, y_pred)` → 5×5 matrix

### S8.4 — Save eval_results.json

```json
{
  "distilbert_finetuned": {
    "accuracy": 0.82,
    "macro_f1": 0.81,
    "per_class": {
      "safe":      { "precision": 0.9, "recall": 0.87, "f1": 0.88 },
      "jailbreak": { "precision": 0.8, "recall": 0.80, "f1": 0.80 },
      "injection": { "precision": 0.85, "recall": 0.80, "f1": 0.82 },
      "toxic":     { "precision": 0.75, "recall": 0.73, "f1": 0.74 },
      "pii":       { "precision": 0.88, "recall": 0.87, "f1": 0.87 }
    }
  }
}
```

(Actual values will differ — this is just the required structure.)

### S8.5 — Confusion matrix plot

Use `seaborn.heatmap` to plot the 5×5 confusion matrix. Label axes with class names. Save to `results/confusion_matrix.png`.

### S8.6 — Keyword baseline (build and evaluate on same test set)

Build a simple rule-based classifier using keyword lists:

```python
KEYWORD_LISTS = {
    "jailbreak": [
        "ignore previous instructions", "pretend you are", "dan mode",
        "jailbreak", "bypass", "no restrictions", "act as if",
        "forget your training", "developer mode", "roleplay as",
    ],
    "injection": [
        "ignore the", "disregard", "system override", "new instruction",
        "ignore all previous", "access_granted", "you are now",
        "injection", "prompt injection", "start your response with",
    ],
    "toxic": [
        "hate", "kill", "stupid", "idiot", "racist", "slur",
        "worthless", "disgusting", "inferior", "swarm",
    ],
    "pii": [
        "social security", "home address", "credit card", "date of birth",
        "passport", "bank account", "medical records", "phone number",
        "personal email", "tax return", "login credentials",
    ],
}

def keyword_predict(text: str) -> str:
    text_lower = text.lower()
    scores = {label: 0 for label in KEYWORD_LISTS}
    for label, keywords in KEYWORD_LISTS.items():
        scores[label] = sum(1 for kw in keywords if kw in text_lower)
    best_label = max(scores, key=scores.get)
    if scores[best_label] == 0:
        return "safe"
    return best_label
```

Evaluate on `test.csv` and compute accuracy + per-class F1. Append to `eval_results.json` under key `"keyword_baseline"`.

### S8.7 — Zero-shot pre-trained baseline

Load `martin-ha/toxic-comment-model` or `unitary/toxic-bert` from HuggingFace Hub.

Note: this is a binary toxicity classifier, not a 5-class classifier. For comparison purposes, map its output to `"toxic"` if it predicts toxic and `"safe"` otherwise. Run on test set. Report binary accuracy (toxic vs non-toxic only). Append to `eval_results.json` under key `"pretrained_baseline"`.

### S8.8 — Accuracy vs strictness curve (save for dashboard)

Sweep threshold values from 0.1 to 0.9 in steps of 0.1. For each threshold:
- A prediction is "blocked" if `max confidence > threshold`
- Measure: recall on harmful prompts (jailbreak + injection + toxic + pii), false positive rate on safe prompts

Save results to `results/threshold_curve.json`:
```json
{
  "thresholds": [0.1, 0.2, ..., 0.9],
  "recall": [0.98, 0.95, ...],
  "fpr": [0.40, 0.25, ...]
}
```

This data feeds the dashboard Component D and PPT Slide 6.

---

## S9 — Phase 7: Error Analysis

**File:** `eval.py` (continuation after S8)  
**Output:** `results/error_analysis.csv`

### S9.1 — Collect all wrong predictions from test set

For every row where `predicted_label != true_label`, record:
- `text`
- `true_label`
- `predicted_label`
- `confidence` (softmax prob of predicted class)
- `true_label_prob` (softmax prob of true class)

### S9.2 — Annotate with failure reason

For each wrong prediction, annotate with one of these reason codes:
- `LENGTH_BIAS` — safe-looking text due to short length; common when short harmful prompt gets predicted as safe
- `CLASS_OVERLAP` — toxic misclassified as jailbreak or vice versa (both adversarial language)
- `INJECTION_JAILBREAK_CONFUSION` — injection misclassified as jailbreak or vice versa
- `SUBTLE_PHRASING` — harmful intent expressed indirectly
- `HIGH_CONFIDENCE_ERROR` — confidence > 0.85 but wrong (most dangerous failure mode)
- `OTHER`

Annotation can be rule-based: if `true_label == "safe" and len(text) < 60` → `LENGTH_BIAS`; if `{true_label, predicted_label} == {"toxic", "jailbreak"}` → `CLASS_OVERLAP`, etc.

### S9.3 — Save error_analysis.csv

Columns: `text, true_label, predicted_label, confidence, true_label_prob, failure_reason`

### S9.4 — Print error summary

Print counts by failure reason. Print the 3 highest-confidence wrong predictions. These are your documented failure cases for PPT Slide 7 and RQ4.

---

## S10 — Phase 8–9: Latency + Baseline Comparison

**File:** `eval.py` (continuation)  
**Output:** `results/latency_report.json`

### S10.1 — Latency measurement protocol

```python
import time
import numpy as np

def measure_latency(model, tokenizer, texts: list[str], n_runs: int = 200):
    """
    texts: pool of prompts to sample from
    Runs n_runs inferences, returns avg and P95 latency in milliseconds.
    """
    # Warm-up: 10 runs not counted
    for _ in range(10):
        sample = texts[0]
        inputs = tokenizer(sample, return_tensors="pt", truncation=True,
                          padding="max_length", max_length=128)
        with torch.no_grad():
            model(**inputs)

    latencies = []
    for i in range(n_runs):
        sample = texts[i % len(texts)]
        inputs = tokenizer(sample, return_tensors="pt", truncation=True,
                          padding="max_length", max_length=128)
        t0 = time.perf_counter()
        with torch.no_grad():
            model(**inputs)
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    return {
        "avg_ms": float(np.mean(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
    }
```

### S10.2 — Measure latency for all three systems

Measure latency for: DistilBERT fine-tuned, keyword baseline, pre-trained baseline.

### S10.3 — Save latency_report.json

```json
{
  "distilbert_finetuned": { "avg_ms": 142, "p95_ms": 198 },
  "keyword_baseline":     { "avg_ms": 0.3, "p95_ms": 0.5 },
  "pretrained_baseline":  { "avg_ms": 165, "p95_ms": 210 }
}
```

(Actual values will differ — this is just the required structure.)

### S10.4 — Build comparison table and print to console

| Model | Accuracy | Macro F1 | P95 Latency |
|---|---|---|---|
| Keyword baseline | ~55% | ~0.42 | <1ms |
| Pre-trained (zero-shot) | ~65% | ~0.52 | ~200ms |
| DistilBERT fine-tuned | ~82% | ~0.81 | ~200ms |

Print this table and save to `results/comparison_table.json`. This is PPT Slide 6 and answers RQ3 and RQ6.

---

## S11 — Red-Team Suite

**File:** `red_team_suite.csv`  
**Must be:** hand-crafted prompts NOT in the training data  
**Total:** 45 prompts

### S11.1 — File schema

Columns: `id, text, true_label, attack_category, notes`

- `id`: integer 1–45
- `text`: the prompt string
- `true_label`: one of `safe, jailbreak, injection, toxic, pii`
- `attack_category`: one of `role_play, dan_style, hypothetical, language_switch, doc_injection, subtle_toxic, direct_pii, benign_coding, benign_general, benign_creative`
- `notes`: short string explaining why this example is interesting or tricky

### S11.2 — Composition requirements

- 10 jailbreak prompts covering: role-play framing, DAN-style, hypothetical wrapping, indirect language
- 10 injection prompts: adversarial text embedded inside a document/email the model is asked to process
- 10 toxic prompts: subtler phrasing than training data, not verbatim copies
- 15 safe/benign prompts: coding, history, science, cooking, creative writing (diverse topics)

### S11.3 — Red-team runner (inside eval.py)

Load `red_team_suite.csv`, run `predict()` on all 45 rows, compute:
- Block rate on red-team prompts (jailbreak + injection + toxic + pii): must be ≥ 80% to pass submission criteria
- False positive rate on benign prompts: must be ≤ 20%

Save per-prompt results (text, true_label, predicted_label, confidence, correct) to `results/red_team_results.csv`.

---

## S12 — Phase 10: Streamlit Dashboard

**File:** `app.py`  
**Run:** `streamlit run app.py`

### S12.1 — Model loading (cached)

```python
import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch, json

@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("./model")
    tokenizer = AutoTokenizer.from_pretrained("./model")
    with open("label2id.json") as f:
        mapping = json.load(f)
    model.eval()
    return model, tokenizer, mapping["id2label"]
```

Do NOT call `load_model()` outside the cache decorator. Without `@st.cache_resource`, the model reloads on every user interaction.

### S12.2 — Component A: Live tester

- `st.text_area("Enter a prompt")` for user input
- `st.slider("Threshold", 0.1, 0.9, 0.5, 0.05)` for threshold
- On submit button click: run inference, display:
  - Verdict: BLOCKED or ALLOWED (color-coded — red for blocked, green for allowed)
  - Category: the predicted class
  - Confidence: `st.progress(confidence)` bar showing 0–1 value
  - Latency: inference time in ms
- If confidence > threshold → BLOCKED, else → ALLOWED

### S12.3 — Component B: Threshold impact demo

Show a read-only panel: for the current input, display what verdict would be at threshold 0.3, 0.5, 0.7. This illustrates the threshold trade-off without requiring a button click.

### S12.4 — Component C: Aggregate metrics panel

Load `results/red_team_results.csv` (pre-computed). Display:
- Total prompts evaluated: 45
- Blocked count + rate
- Missed count + rate
- False positive count + rate on benign set
- Per-class recall table (jailbreak, injection, toxic, pii)

Use `st.metric()` for the top-level numbers.

### S12.5 — Component D: Accuracy vs strictness curve

Load `results/threshold_curve.json`. Plot recall and FPR as two lines on a Plotly line chart against threshold (x-axis 0.1–0.9). Add a vertical line at the deployed threshold value.

```python
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=thresholds, y=recall, name="Recall (harmful)"))
fig.add_trace(go.Scatter(x=thresholds, y=fpr, name="False positive rate"))
fig.add_vline(x=0.5, line_dash="dash", annotation_text="deployed threshold")
st.plotly_chart(fig)
```

### S12.6 — Page layout

```python
st.set_page_config(page_title="SmartGuard", layout="wide")
st.title("SmartGuard — LLM Prompt Firewall")

tab1, tab2, tab3 = st.tabs(["Live tester", "Aggregate metrics", "Threshold curve"])
```

---

## S13 — Supporting Files

### S13.1 — label2id.json

Generated by `train.py` during setup (S6.1). Must be committed to repo.

```json
{
  "label2id": { "safe": 0, "jailbreak": 1, "injection": 2, "toxic": 3, "pii": 4 },
  "id2label": { "0": "safe", "1": "jailbreak", "2": "injection", "3": "toxic", "4": "pii" }
}
```

### S13.2 — Loss curve plot

Generated at end of `train.py`. Load `results/training_log.csv`, plot `train_loss` and `eval_loss` vs `epoch` using matplotlib. Save to `results/loss_curve.png`. This is PPT Slide 11.

```python
import matplotlib.pyplot as plt
import pandas as pd

log = pd.read_csv("results/training_log.csv")
plt.figure(figsize=(8, 5))
plt.plot(log["epoch"], log["train_loss"], label="Training loss", marker="o")
plt.plot(log["epoch"], log["eval_loss"], label="Validation loss", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs validation loss")
plt.savefig("results/loss_curve.png", dpi=150, bbox_inches="tight")
```

### S13.3 — README.md required sections

The README must include all of the following in this order:

1. **Problem statement** — why keyword filters are insufficient
2. **Track choice** — Track B selected, with justification: size/speed/accuracy trade-off of DistilBERT vs alternatives
3. **Dataset** — sources (JBB-Behaviors, BIPIA, Nemotron-PII, ToxiGen, Civil Comments, Alpaca-Cleaned), class distribution table, known biases (Civil Comments identity term bias, length disparity between classes)
4. **Model architecture** — DistilBERT-base-uncased, 66M parameters, 6 transformer layers, classification head
5. **Training setup** — epochs, batch size, learning rate, early stopping, hardware, total training time
6. **Results** — comparison table (keyword / pre-trained / fine-tuned), P95 latency result (explicit number)
7. **Error analysis** — minimum 3 failure patterns with example texts and explanation
8. **Setup instructions** — must run in ≤ 5 commands:
   ```bash
   git clone <repo>
   cd smartguard
   pip install -r requirements.txt
   python prepare_data.py
   python train.py
   python eval.py
   streamlit run app.py
   ```

---

## S14 — Research Questions Mapping

These are graded directly. Every answer must appear in both the README and the PPT.

| RQ | Answer location | Source data |
|---|---|---|
| RQ1: Does model beat keyword filter? | README §7, PPT Slide 5 | `eval_results.json` — compare keyword vs DistilBERT F1 + 3–5 side-by-side examples from error analysis |
| RQ2: Accuracy vs strictness trade-off | README §6, PPT Slide 6 | `threshold_curve.json` + state chosen deployment threshold and justify it |
| RQ3: P95 latency on CPU | README §5, PPT Slide 2 | `latency_report.json` — state exact milliseconds, state whether <100ms real-time threshold is met |
| RQ4: Where does system fail? | README §7, PPT Slide 7 | `error_analysis.csv` — 3–5 documented failure cases with linguistic pattern explanation |
| RQ5: What to improve next? | README §8, PPT Slide 8 | Write one specific paragraph: "If I had 2 more days..." — must name a specific linguistic failure |
| RQ6 (Track B): Did training outperform pre-trained baseline? | README §6, PPT Slide 11 | `eval_results.json` — compare fine-tuned vs pretrained_baseline F1 per class |
| RQ7 (Track B): Loss curve analysis | README §5, PPT Slide 11 | `loss_curve.png` + `training_log.csv` — state epoch where divergence appeared, whether early stopping triggered |

---

## S15 — PPT Slide Mapping

11 slides required for Track B. Map each slide to its data source:

| Slide | Title | Data source |
|---|---|---|
| 1 | Problem: why keyword filters fail | Qualitative + 2–3 failure examples from keyword baseline |
| 2 | Track B: model choice + justification | DistilBERT specs: 66M params, CPU P95 latency number from `latency_report.json` |
| 3 | System architecture: 4 components | Diagram: Classifier → Threshold Engine → Red-Team Runner → Dashboard |
| 4 | Red-team results: per-category recall | `red_team_results.csv` — table of recall per class + overall block rate |
| 5 | Keyword filter vs DistilBERT examples | 3–5 rows from `error_analysis.csv` showing keyword failure + model success |
| 6 | Accuracy vs strictness curve | Plot from `threshold_curve.json` — Plotly or matplotlib export |
| 7 | Failure cases (highest weight slide) | 3–5 rows from `error_analysis.csv` with failure reason + explanation |
| 8 | What we'd improve next | One specific paragraph — tied to RQ5 |
| 9 | Dataset composition | `dataset_stats.json` — class distribution bar chart, text length per class |
| 10 | Training setup | Hyperparameters table + training time from `train.py` output |
| 11 | Loss curves + model comparison | `loss_curve.png` + comparison table from `eval_results.json` |

---

## S16 — Final Checklist

Before submission, verify every item:

**Code and files**
- [ ] `prepare_data.py` runs without errors, outputs `final_dataset.csv`, `train.csv`, `val.csv`, `test.csv`
- [ ] `train.py` runs without errors, outputs `model/` directory and `training_log.csv`
- [ ] `eval.py` runs without errors, outputs `eval_results.json`, `confusion_matrix.png`, `error_analysis.csv`, `latency_report.json`, `red_team_results.csv`, `threshold_curve.json`
- [ ] `app.py` runs with `streamlit run app.py`, all 4 components functional
- [ ] `red_team_suite.csv` committed with 45 rows and ground-truth labels
- [ ] `label2id.json` committed
- [ ] `requirements.txt` has pinned versions and random_state=42 comment
- [ ] `results/loss_curve.png` committed
- [ ] `results/comparison_table.json` committed

**Research requirements**
- [ ] All 7 Research Questions answered in README
- [ ] Error analysis section has ≥ 3 failure patterns with example texts
- [ ] P95 latency stated as explicit number in README
- [ ] Track B justification paragraph in README
- [ ] Loss curve shows both training and validation loss

**Evaluation targets (graded criteria)**
- [ ] Block rate on red-team harmful prompts ≥ 80%
- [ ] False positive rate on benign prompts ≤ 20%
- [ ] All 3 attack categories covered (jailbreak, injection, toxic)
- [ ] At least 1 accuracy vs strictness curve committed

**PPT**
- [ ] 11 slides (Slides 9–11 are Track B mandatory)
- [ ] Slide 7 (failure cases) has ≥ 3 concrete examples with explanations
- [ ] Slide 11 (loss curves) shows training vs validation loss with overfitting discussion

**Demo video**
- [ ] Shows safe prompt passing with low confidence score
- [ ] Shows jailbreak being blocked with category and score visible
- [ ] Shows one real failure case (miss or false positive) with explanation
- [ ] Dashboard visible with aggregate metrics at some point in video
- [ ] Do NOT hide failures — showing a real miss is required

---

## APPENDIX A — Critical implementation warnings

1. **Do NOT use plain `pd.read_csv()` on the raw files** — they have 7 blank header rows. Use the `parse_smartguard_csv()` function defined in S4.1.

2. **Do NOT use `.cuda()` or GPU anywhere** — submission must run on CPU only. Set `no_cuda=True` in TrainingArguments and use `device = torch.device("cpu")` everywhere.

3. **Do NOT look at test.csv before final eval** — all hyperparameter decisions must use val set only.

4. **Do NOT forget `@st.cache_resource`** on the model loading function in app.py — without it the app reloads the model on every user interaction.

5. **The `split` column in source CSVs is from the original data source, not your split** — ignore it and do your own stratified 70/15/15 split.

6. **Red-team suite must be new prompts** — do not copy rows from the 500-row training dataset into `red_team_suite.csv`.

7. **Warm up the model before measuring latency** — run 10 dummy inferences before starting the timer. PyTorch caches JIT compilation on the first run, which skews measurements.

8. **Overfitting is expected and should be documented, not fought** — with 350 training rows and 66M parameters, the validation loss will likely rise after epoch 2–3. Early stopping handles it. Document the epoch where divergence appears in the README and PPT as a research finding.

---

## APPENDIX B — Expected output values (approximate)

These are estimates based on dataset size and model. Actual values will differ.

| Metric | Expected range |
|---|---|
| DistilBERT test accuracy | 75–88% |
| DistilBERT macro F1 | 0.74–0.87 |
| Keyword baseline accuracy | 45–65% |
| Keyword baseline macro F1 | 0.35–0.55 |
| DistilBERT P95 CPU latency | 80–250ms (hardware dependent) |
| Keyword P95 CPU latency | <2ms |
| Epoch where val loss diverges | 2–4 (dependent on data) |
| Best epoch (early stopping) | 3–5 |
| Training time (CPU) | 3–8 minutes |
| Red-team block rate | Should hit ≥80% |
| Red-team false positive rate | Should be ≤20% |
