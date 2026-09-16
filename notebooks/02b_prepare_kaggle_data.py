import os
import sys
import pandas as pd

# Allow importing from src/
sys.path.append(os.path.abspath('..'))
from src.preprocess import clean_text

print("Environment ready for large dataset processing!")

# 1. Load data
raw_path = '../data/raw/train.csv'
df_kaggle = pd.read_csv(raw_path)
print(f"Dataset loaded: {len(df_kaggle)} rows!")
print(f"Detected columns: {df_kaggle.columns.tolist()}\n")

# 2. Identify text column
text_candidates = ['text', 'tweet', 'headline', 'title', 'content', 'statement']
text_col = next((col for col in text_candidates if col in df_kaggle.columns), None)

if text_col is None:
    raise ValueError(f"Could not find a text column! Available columns: {df_kaggle.columns.tolist()}")

print(f"Using '{text_col}' as the text column.")

# 3. Identify label column
label_candidates = ['label_encoded', 'target', 'label', 'class', 'category', 'fake', 'is_fake']
label_col = next((col for col in label_candidates if col in df_kaggle.columns), None)

if label_col is None:
    raise ValueError(f"Could not find a label column! Available columns: {df_kaggle.columns.tolist()}")

print(f"Using '{label_col}' as the label column.")

# 4. Standardize labels to 0 and 1
# Check unique values in the label column
unique_vals = df_kaggle[label_col].dropna().unique()
print(f"Unique values in '{label_col}': {unique_vals}")

label_mapping = {
    'real': 0, 'real ': 0, 'true': 0, 'TRUE': 0, '0': 0, 0: 0,
    'fake': 1, 'fake ': 1, 'false': 1, 'FALSE': 1, '1': 1, 1: 1
}

# If it's already 0/1 numeric, keep it; otherwise map strings
if df_kaggle[label_col].dtype in ['int64', 'float64'] and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
    df_kaggle['label_encoded'] = df_kaggle[label_col].astype(int)
else:
    df_kaggle['label_encoded'] = df_kaggle[label_col].astype(str).str.strip().str.lower().map(label_mapping)

# 5. Clean text
print("\nCleaning text... this may take 1-2 minutes for ~95k rows...")
df_kaggle['cleaned_text'] = df_kaggle[text_col].astype(str).apply(clean_text)

# 6. Drop missing/empty rows
initial_count = len(df_kaggle)
df_kaggle = df_kaggle.dropna(subset=['cleaned_text', 'label_encoded'])
df_kaggle = df_kaggle[df_kaggle['cleaned_text'].str.strip() != '']
df_kaggle['label_encoded'] = df_kaggle['label_encoded'].astype(int)
print(f"Retained {len(df_kaggle)} valid rows out of {initial_count}.")

# 7. Save processed dataset
output_path = '../data/processed/large_cleaned_disaster_news.csv'
os.makedirs('../data/processed', exist_ok=True)
df_kaggle[['cleaned_text', 'label_encoded']].to_csv(output_path, index=False)

print(f"\nSuccess! Cleaned data saved to: {output_path}")
print(df_kaggle[['cleaned_text', 'label_encoded']].head())