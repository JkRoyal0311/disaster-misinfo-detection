import pandas as pd
import sys
import os

# Allow the notebook to import from the src/ folder
sys.path.append(os.path.abspath('..'))
from src.preprocess import clean_text

print("Environment ready for large dataset processing!")
# Load the Kaggle dataset
df_kaggle = pd.read_csv('../data/raw/train.csv')

print(f"Massive Dataset Loaded: {len(df_kaggle)} rows!")
print("\nFirst 5 rows of raw data:")
df_kaggle.head()
# 1. Standardize column names for our pipeline
# Kaggle usually uses 'target' (1=disaster/real, 0=not disaster/fake)
# We will rename them so Week 3's code knows exactly where to look.
if 'target' in df_kaggle.columns:
    df_kaggle = df_kaggle.rename(columns={'target': 'label_encoded'})

# 2. Apply our reusable text cleaner from Week 1 to the 'text' column
print("Cleaning thousands of tweets... this might take a few seconds...")
df_kaggle['cleaned_text'] = df_kaggle['text'].apply(clean_text)

# 3. Drop any rows where text might have become completely empty after cleaning
df_kaggle = df_kaggle.dropna(subset=['cleaned_text', 'label_encoded'])
df_kaggle = df_kaggle[df_kaggle['cleaned_text'].str.strip() != '']

# 4. Save this massive, clean dataset for Week 3
output_path = '../data/processed/large_cleaned_disaster_news.csv'
os.makedirs('../data/processed', exist_ok=True)
df_kaggle[['cleaned_text', 'label_encoded']].to_csv(output_path, index=False)

print(f"\nSuccess! Cleaned data saved to: {output_path}")
df_kaggle[['cleaned_text', 'label_encoded']].head()