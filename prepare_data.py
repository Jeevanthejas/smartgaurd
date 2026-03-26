import pandas as pd
import numpy as np
import json
import re
from io import StringIO
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

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
    data_lines = [line.rstrip('\\\n').rstrip('\\\r') + '\n' for line in data_lines]

    # Strip RTF prefix from header line (present in PII.csv)
    data_lines[0] = re.sub(r'^.*?id,', 'id,', data_lines[0])

    content = ''.join(data_lines)
    df = pd.read_csv(StringIO(content))
    return df

FILE_MAP = {
    'jailbreak': 'data/raw/jailbreak.csv',
    'injection': 'data/raw/indirect.csv',
    'pii':       'data/raw/PII.csv',
    'toxic':     'data/raw/toxic.csv',
    'safe':      'data/raw/safe.csv',
}

CUSTOM_DATA_PATH = 'data/raw/custom.csv'

def main():
    print("=" * 60)
    print("SmartGuard — Phase 1 & 2: Data Preparation + Stratified Split")
    print("=" * 60)

    print("\nLoading and parsing raw CSV files...")
    dfs = []
    
    for label, path in FILE_MAP.items():
        df_part = parse_smartguard_csv(path)
        if 'text' in df_part.columns and 'label' in df_part.columns:
            df_part = df_part[['text', 'label']]
        else:
            raise KeyError(f"'text' and/or 'label' columns not found in {path}")
        dfs.append(df_part)

    # Load custom user-added prompts if the file exists
    import os
    if os.path.exists(CUSTOM_DATA_PATH):
        print(f"\n📂 Loading custom prompts from {CUSTOM_DATA_PATH}...")
        df_custom = pd.read_csv(CUSTOM_DATA_PATH)
        if 'text' in df_custom.columns and 'label' in df_custom.columns:
            df_custom = df_custom[['text', 'label']]
            print(f"   Found {len(df_custom)} custom prompts")
            print(f"   Custom label distribution:\n{df_custom['label'].value_counts().to_string()}")
            dfs.append(df_custom)
        else:
            print(f"   ⚠️ Skipping custom.csv — missing 'text' or 'label' columns")

    df = pd.concat(dfs, ignore_index=True)
    
    # Validate labels
    valid_labels = {'safe', 'jailbreak', 'injection', 'toxic', 'pii'}
    assert set(df['label'].unique()).issubset(valid_labels), f"Found invalid labels: {df['label'].unique()}"
    
    print("\nValue counts before cleaning:")
    print(df['label'].value_counts())
    
    initial_count = len(df)
    
    # 1. Drop rows where text is null or NaN
    df = df.dropna(subset=['text'])
    # 2. Strip leading/trailing whitespace
    df['text'] = df['text'].str.strip()
    # 3. Collapse internal whitespace
    df['text'] = df['text'].astype(str).str.replace(r'\s+', ' ', regex=True)
    # 4. Drop empty strings
    df = df[df['text'] != '']
    # 5. Drop short texts
    df = df[df['text'].str.len() >= 10]
    # 6. Drop duplicates
    df = df.drop_duplicates(subset=['text'], keep='first')
    # 7. Reset index
    df = df.reset_index(drop=True)
    
    final_count = len(df)
    print(f"\nRows before cleaning: {initial_count}, Rows after cleaning: {final_count}, Delta: {initial_count - final_count}")
    
    # Exploratory stats
    stats = {
        "total_rows_after_cleaning": final_count,
        "rows_per_class": dict(df['label'].value_counts()),
        "mean_text_length_per_class": dict(df.groupby('label')['text'].apply(lambda x: x.str.len().mean())),
        "min_text_length_per_class": dict(df.groupby('label')['text'].apply(lambda x: x.str.len().min())),
        "max_text_length_per_class": dict(df.groupby('label')['text'].apply(lambda x: x.str.len().max())),
    }
    
    for k, v in stats.items():
        if isinstance(v, dict):
            stats[k] = {str(k2): int(v2) if isinstance(v2, (np.int64, np.integer)) else float(v2) for k2, v2 in v.items()}
    
    df.to_csv('data/processed/final_dataset.csv', index=False)
    
    # ---- Phase 2: Stratified Split ----
    print("\n" + "=" * 60)
    print("Phase 2: Stratified Split (70/15/15)")
    print("=" * 60)
    
    X = df['text']
    y = df['label']
    
    # Step 1: split off 15% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_SEED
    )
    
    # Step 2: split remaining 85% into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.176, stratify=y_trainval, random_state=RANDOM_SEED
    )
    
    pd.DataFrame({'text': X_train, 'label': y_train}).to_csv('data/processed/train.csv', index=False)
    pd.DataFrame({'text': X_val,   'label': y_val  }).to_csv('data/processed/val.csv',   index=False)
    
    # TEST SET LOCKED — do not use for any decisions until final evaluation in eval.py
    pd.DataFrame({'text': X_test,  'label': y_test }).to_csv('data/processed/test.csv',  index=False)
    
    print("\nSplit Distribution:")
    print(f"  Train: {len(X_train)} rows")
    print(y_train.value_counts().to_string())
    print(f"\n  Val: {len(X_val)} rows")
    print(y_val.value_counts().to_string())
    print(f"\n  Test: {len(X_test)} rows")
    print(y_test.value_counts().to_string())
    
    stats["train_size"] = len(X_train)
    stats["val_size"] = len(X_val)
    stats["test_size"] = len(X_test)
    stats["train_per_class"] = {str(k): int(v) for k, v in y_train.value_counts().to_dict().items()}
    stats["val_per_class"] = {str(k): int(v) for k, v in y_val.value_counts().to_dict().items()}
    stats["test_per_class"] = {str(k): int(v) for k, v in y_test.value_counts().to_dict().items()}
    
    with open('data/processed/dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=4)
        
    print("\n✅ Dataset preparation completed successfully.")

if __name__ == "__main__":
    main()
