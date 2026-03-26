import pandas as pd
import numpy as np
import json
import time
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

device = torch.device("cpu")

with open('label2id.json', 'r') as f:
    mapping = json.load(f)
    ID2LABEL = {int(k): v for k, v in mapping["id2label"].items()}

def load_test_data():
    return pd.read_csv('data/processed/test.csv')

def load_red_team_data():
    return pd.read_csv('red_team_suite.csv')

def main():
    print("=" * 60)
    print("SmartGuard — Phase 6–9: Evaluation, Error Analysis, Latency")
    print("=" * 60)

    print("\nLoading model for evaluation...")
    model = AutoModelForSequenceClassification.from_pretrained("./model")
    tokenizer = AutoTokenizer.from_pretrained("./model")
    model.eval()
    model.to(device)

    def predict(texts: list[str], threshold: float = 0.5):
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

    # ---- Phase 6: Test Set Evaluation ----
    test_df = load_test_data()
    y_true = test_df['label'].tolist()
    texts = test_df['text'].tolist()

    print("Running inference on test set...")
    preds = predict(texts)
    y_pred = [p[0] for p in preds]
    y_conf = [p[1] for p in preds]
    all_probs_list = [p[2] for p in preds]

    clf_report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(ID2LABEL.values()))

    eval_results = {
        "distilbert_finetuned": {
            "accuracy": clf_report["accuracy"],
            "macro_f1": clf_report["macro avg"]["f1-score"],
            "per_class": {
                label: {
                    "precision": clf_report[label]["precision"],
                    "recall": clf_report[label]["recall"],
                    "f1": clf_report[label]["f1-score"]
                }
                for label in ID2LABEL.values() if label in clf_report
            }
        }
    }

    # Confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=list(ID2LABEL.values()), yticklabels=list(ID2LABEL.values()), cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("results/confusion_matrix.png", dpi=150, bbox_inches="tight")
    print("✅ Saved confusion matrix.")

    # ---- Keyword Baseline ----
    print("\nEvaluating Keyword Baseline...")
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

    y_pred_kw = [keyword_predict(t) for t in texts]
    kw_report = classification_report(y_true, y_pred_kw, output_dict=True, zero_division=0)
    eval_results["keyword_baseline"] = {
        "accuracy": kw_report["accuracy"],
        "macro_f1": kw_report["macro avg"]["f1-score"],
        "per_class": {
            label: {
                "precision": kw_report.get(label, {}).get("precision", 0),
                "recall": kw_report.get(label, {}).get("recall", 0),
                "f1": kw_report.get(label, {}).get("f1-score", 0)
            }
            for label in ID2LABEL.values()
        }
    }

    # ---- Pre-trained Baseline ----
    print("Evaluating Pretrained Baseline...")
    toxic_pipeline = pipeline("text-classification", model="martin-ha/toxic-comment-model", device=-1)

    y_pred_pt = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = [t[:512] for t in texts[i:i+batch_size]]
        outputs = toxic_pipeline(batch)
        for out in outputs:
            if out['label'].lower() == 'toxic':
                y_pred_pt.append('toxic')
            else:
                y_pred_pt.append('safe')

    y_true_binary = ['toxic' if l == 'toxic' else 'safe' for l in y_true]
    y_pred_binary = ['toxic' if l == 'toxic' else 'safe' for l in y_pred_pt]
    pt_acc = accuracy_score(y_true_binary, y_pred_binary)
    pt_f1 = f1_score(y_true_binary, y_pred_binary, pos_label='toxic', average="binary") if "toxic" in y_true_binary else 0.0

    eval_results["pretrained_baseline"] = {
        "accuracy": pt_acc,
        "f1_toxic_class": pt_f1,
        "note": "Evaluated only as binary toxic/safe"
    }

    with open('results/eval_results.json', 'w') as f:
        json.dump(eval_results, f, indent=4)

    # ---- Threshold Curve ----
    print("Computing Threshold Curve...")
    thresholds = [round(x, 1) for x in np.arange(0.1, 1.0, 0.1)]
    recall_arr, fpr_arr = [], []

    for t in thresholds:
        harmful_total, harmful_blocked = 0, 0
        safe_total, safe_blocked = 0, 0

        for i in range(len(y_true)):
            pred_l = y_pred[i]
            conf = y_conf[i]
            is_blocked = (conf > t) and (pred_l != 'safe')

            if y_true[i] == 'safe':
                safe_total += 1
                if is_blocked:
                    safe_blocked += 1
            else:
                harmful_total += 1
                if is_blocked:
                    harmful_blocked += 1

        recall_arr.append(harmful_blocked / harmful_total if harmful_total else 0)
        fpr_arr.append(safe_blocked / safe_total if safe_total else 0)

    with open('results/threshold_curve.json', 'w') as f:
        json.dump({"thresholds": thresholds, "recall": recall_arr, "fpr": fpr_arr}, f, indent=4)

    # ---- Phase 7: Error Analysis ----
    print("\nError Analysis...")
    error_rows = []
    for i in range(len(y_true)):
        t = texts[i]
        tl = y_true[i]
        pl = y_pred[i]
        c = y_conf[i]

        if tl != pl:
            tl_prob = all_probs_list[i].get(tl, 0.0)

            reason = "OTHER"
            if tl == 'safe' and len(t) < 60:
                reason = "LENGTH_BIAS"
            elif {tl, pl} == {"toxic", "jailbreak"}:
                reason = "CLASS_OVERLAP"
            elif {tl, pl} == {"injection", "jailbreak"}:
                reason = "INJECTION_JAILBREAK_CONFUSION"
            elif c > 0.85:
                reason = "HIGH_CONFIDENCE_ERROR"
            elif tl != 'safe' and pl == 'safe':
                reason = "SUBTLE_PHRASING"

            error_rows.append({
                "text": t, "true_label": tl, "predicted_label": pl,
                "confidence": c, "true_label_prob": tl_prob, "failure_reason": reason
            })

    err_df = pd.DataFrame(error_rows)
    err_df.to_csv("results/error_analysis.csv", index=False)
    if not err_df.empty:
        print("Failure summary:")
        print(err_df['failure_reason'].value_counts())
        print("\nTop 3 Highest Confidence Wrongs:")
        for _, r in err_df.sort_values(by="confidence", ascending=False).head(3).iterrows():
            print(f"  True: {r['true_label']} | Pred: {r['predicted_label']} ({r['confidence']:.2f}) | {r['text'][:80]}")
    else:
        print("No errors on test set.")

    # ---- Phase 8–9: Latency ----
    print("\nMeasuring Latency...")

    def measure_latency(runner_fn, texts_pool, n_runs=200, is_pt=False, is_kw=False):
        sample = texts_pool[0]
        for _ in range(10):
            if is_pt:
                runner_fn([sample[:512]])
            elif is_kw:
                runner_fn(sample)
            else:
                inputs = tokenizer(sample, return_tensors="pt", truncation=True, padding="max_length", max_length=128).to(device)
                with torch.no_grad():
                    runner_fn(**inputs)

        latencies = []
        for i in range(n_runs):
            sample = texts_pool[i % len(texts_pool)]
            t0 = time.perf_counter()
            if is_pt:
                runner_fn([sample[:512]])
            elif is_kw:
                runner_fn(sample)
            else:
                inputs = tokenizer(sample, return_tensors="pt", truncation=True, padding="max_length", max_length=128).to(device)
                with torch.no_grad():
                    runner_fn(**inputs)
            latencies.append((time.perf_counter() - t0) * 1000)

        return {
            "avg_ms": float(np.mean(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
        }

    lat_distil = measure_latency(model, texts)
    lat_kw = measure_latency(keyword_predict, texts, is_kw=True)
    lat_pt = measure_latency(toxic_pipeline, texts, is_pt=True)

    latency_report = {
        "distilbert_finetuned": lat_distil,
        "keyword_baseline": lat_kw,
        "pretrained_baseline": lat_pt
    }
    with open('results/latency_report.json', 'w') as f:
        json.dump(latency_report, f, indent=4)

    comp_table = [
        {"Model": "Keyword baseline", "Accuracy": round(kw_report['accuracy'], 2), "Macro F1": round(kw_report['macro avg']['f1-score'], 2), "P95 Latency": f"{round(lat_kw['p95_ms'], 1)}ms"},
        {"Model": "Pre-trained (zero-shot)", "Accuracy": round(pt_acc, 2), "Macro F1": round(pt_f1, 2), "P95 Latency": f"{round(lat_pt['p95_ms'], 1)}ms"},
        {"Model": "DistilBERT fine-tuned", "Accuracy": round(clf_report['accuracy'], 2), "Macro F1": round(clf_report['macro avg']['f1-score'], 2), "P95 Latency": f"{round(lat_distil['p95_ms'], 1)}ms"},
    ]
    with open('results/comparison_table.json', 'w') as f:
        json.dump(comp_table, f, indent=4)
    print("\nComparison Table:")
    for r in comp_table:
        print(f"  {r}")

    # ---- Red-Team Evaluation ----
    print("\nRed-Team Evaluation...")
    rt_df = load_red_team_data()
    rt_texts = rt_df['text'].tolist()
    rt_labels = rt_df['true_label'].tolist()

    preds_rt = predict(rt_texts)

    rt_results = []
    harmful_total, harmful_blocked = 0, 0
    benign_total, benign_fp = 0, 0

    for i in range(len(rt_labels)):
        pl = preds_rt[i][0]
        c = preds_rt[i][1]
        tl = rt_labels[i]
        is_blocked = (c > 0.5) and (pl != 'safe')
        correct = (is_blocked and tl != 'safe') or (not is_blocked and tl == 'safe')

        rt_results.append({
            "text": rt_texts[i], "true_label": tl, "predicted_label": pl,
            "confidence": c, "correct": correct
        })

        if tl == 'safe':
            benign_total += 1
            if is_blocked:
                benign_fp += 1
        else:
            harmful_total += 1
            if is_blocked:
                harmful_blocked += 1

    pd.DataFrame(rt_results).to_csv("results/red_team_results.csv", index=False)

    print(f"\n  Red-team Block Rate (Harmful): {harmful_blocked/harmful_total*100:.1f}%")
    if benign_total > 0:
        print(f"  Red-team FPR (Benign):         {benign_fp/benign_total*100:.1f}%")
    print("\n✅ Evaluation complete.")

if __name__ == "__main__":
    main()
