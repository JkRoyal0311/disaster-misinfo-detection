# Generated from: 03_bert_model.ipynb
# Converted at: 2026-08-31T10:18:45.837Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import pandas as pd
import torch
import os
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

# Load the cleaned dataset
df = pd.read_csv('../data/processed/cleaned_disaster_news.csv')
df = df.dropna(subset=['cleaned_text', 'label_encoded'])

# Split into Train and Validation sets
texts = df['cleaned_text'].tolist()
labels = df['label_encoded'].tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

print(f"Train size: {len(train_texts)}, Validation size: {len(val_texts)}")

# Load the DistilBERT tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Tokenize the text
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# Create a custom PyTorch Dataset class
class DisasterDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = DisasterDataset(train_encodings, train_labels)
val_dataset = DisasterDataset(val_encodings, val_labels)

print("Data tokenized and formatted for PyTorch!")

# Function to calculate metrics during training
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}

# Load the pre-trained DistilBERT model (with 2 output labels)
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Set training arguments
# Set training arguments
training_args = TrainingArguments(
    output_dir='./results',          
    num_train_epochs=3,              
    per_device_train_batch_size=8,   
    per_device_eval_batch_size=8,    
    warmup_steps=10,                 
    weight_decay=0.01,               
    logging_dir='./logs',            
    logging_steps=5,
    eval_strategy="epoch"      # Updated parameter name!
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Start training
print("Starting training...")
trainer.train()

# Save the fine-tuned model and tokenizer
model_path = "../models/distilbert_disaster"
os.makedirs(model_path, exist_ok=True)

model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

print(f"\nSuccess: Fine-tuned DistilBERT saved to {model_path}")