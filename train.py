import time
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score

RANDOM_SEED = 42

LABEL2ID = {
    "safe":      0,
    "jailbreak": 1,
    "injection": 2,
    "toxic":     3,
    "pii":       4,
}
ID2LABEL = {str(v): k for k, v in LABEL2ID.items()}
ID2LABEL_INT = {v: k for k, v in LABEL2ID.items()}

MODEL_NAME = "distilbert-base-uncased"

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="macro")),
    }

def main():
    print("=" * 60)
    print("SmartGuard — Phase 3–5: Encoding, Tokenization & Training")
    print("=" * 60)

    # Save label mapping
    with open('label2id.json', 'w') as f:
        json.dump({"label2id": LABEL2ID, "id2label": ID2LABEL}, f, indent=2)
    print("Saved label2id.json")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    def load_split(path):
        df = pd.read_csv(path)
        df['labels'] = df['label'].map(LABEL2ID)
        ds = Dataset.from_pandas(df[['text', 'labels']])
        return ds.map(tokenize, batched=True)

    print("\nLoading datasets...")
    train_ds = load_split('data/processed/train.csv')
    val_ds   = load_split('data/processed/val.csv')
    test_ds  = load_split('data/processed/test.csv')

    # Verification step
    print("\nVerification — one example per class:")
    seen_classes = set()
    for ex in train_ds:
        lbl = ex['labels']
        if lbl not in seen_classes:
            seen_classes.add(lbl)
            decoded = tokenizer.decode(ex['input_ids'], skip_special_tokens=True)
            print(f"  Class {ID2LABEL_INT[lbl]:>10s} ({lbl}) → {decoded[:80]}...")
            if len(seen_classes) == 5:
                break

    print("\nLoading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=5,
        id2label=ID2LABEL_INT,
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir="./model",
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        warmup_steps=100,
        weight_decay=0.01,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        seed=RANDOM_SEED,
        no_cuda=True,           # CPU only — mandatory
        report_to="none",
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    print("\nStarting training...")
    start = time.time()
    trainer.train()
    training_time_seconds = time.time() - start
    print(f"\n⏱ Training time: {training_time_seconds:.1f}s")

    print("Saving model and tokenizer...")
    trainer.save_model("./model")
    tokenizer.save_pretrained("./model")

    # Extract training log
    history = trainer.state.log_history
    epoch_dict = {}
    for entry in history:
        ep = entry.get('epoch')
        if ep is None:
            continue
        ep = round(ep, 2)
        if ep not in epoch_dict:
            epoch_dict[ep] = {'epoch': int(ep)}
        if 'loss' in entry:
            epoch_dict[ep]['train_loss'] = entry['loss']
        if 'eval_loss' in entry:
            epoch_dict[ep]['eval_loss'] = entry['eval_loss']
        if 'eval_accuracy' in entry:
            epoch_dict[ep]['eval_accuracy'] = entry['eval_accuracy']
        if 'eval_f1' in entry:
            epoch_dict[ep]['eval_f1'] = entry['eval_f1']

    int_epoch_dict = {}
    for ep, data in epoch_dict.items():
        ep_int = int(round(ep))
        if ep_int not in int_epoch_dict:
            int_epoch_dict[ep_int] = {'epoch': ep_int}
        int_epoch_dict[ep_int].update(data)

    log_df = pd.DataFrame(list(int_epoch_dict.values()))
    if 'train_loss' in log_df.columns:
        log_df['train_loss'] = log_df['train_loss'].ffill()
    log_df = log_df.dropna(subset=['eval_loss'])
    log_df.to_csv("results/training_log.csv", index=False)

    print("\nFinal validation metrics (best checkpoint):")
    metrics = trainer.evaluate()
    print(f"  Eval Accuracy: {metrics.get('eval_accuracy', 'N/A')}")
    print(f"  Eval F1:       {metrics.get('eval_f1', 'N/A')}")

    best_epoch, best_f1 = None, 0.0
    for _, row in log_df.iterrows():
        if row.get('eval_f1', 0) > best_f1:
            best_f1 = row['eval_f1']
            best_epoch = row['epoch']
    print(f"  Best epoch:    {best_epoch}")

    # Loss curve
    plt.figure(figsize=(8, 5))
    if 'train_loss' in log_df.columns and not log_df['train_loss'].isna().all():
        plt.plot(log_df["epoch"], log_df["train_loss"], label="Training loss", marker="o")
    if 'eval_loss' in log_df.columns:
        plt.plot(log_df["epoch"], log_df["eval_loss"], label="Validation loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs validation loss")
    plt.savefig("results/loss_curve.png", dpi=150, bbox_inches="tight")
    print("\n✅ Saved results/loss_curve.png")

if __name__ == "__main__":
    main()
