# 🚨 AI-Powered Fake News & Misinformation Detection during Climate Disasters

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face: Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E.svg?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![UI: Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end NLP and Deep Learning pipeline designed to identify, filter, and flag rumors, fake alerts, and misinformation circulating across social media and digital platforms during severe climate emergencies (floods, cyclones, earthquakes, and landslides).

---

## 📌 Table of Contents

- [Overview & Problem Statement](#-overview--problem-statement)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Data Pipeline & Preprocessing](#-data-pipeline--preprocessing)
- [Model Training & Evaluation](#-model-training--evaluation)
  - [1. Baseline Model (TF-IDF + Logistic Regression)](#1-baseline-model-tf-idf--logistic-regression)
  - [2. Deep Learning Model (DistilBERT Fine-Tuning)](#2-deep-learning-model-distilbert-fine-tuning)
- [Running the Interactive Web App](#-running-the-interactive-web-app)
- [Example Predictions](#-example-predictions)
- [Roadmap & Future Improvements](#-roadmap--future-improvements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌪 Overview & Problem Statement

During catastrophic climate disasters, social media channels become critical lifelines for real-time SOS requests, evacuation notices, and resource distribution. However, they simultaneously become breeding grounds for:

* **Fabricated alerts and false predictions** that trigger mass hysteria and stampedes.
* **Fraudulent donation links and scam campaigns** diverting emergency aid.
* **Manipulated images and outdated media** that confuse search-and-rescue teams.

**Objective:** Build an open-source, reproducible text classification framework that parses incoming headlines and tweets, neutralizes noise, and accurately predicts whether disaster-related content is **Real News (0)** or **Fake News / Rumor (1)** with high confidence.

---

## ✨ Key Features

* **Modular Text Preprocessing Engine (`src/preprocess.py`):** Strips URLs, user mentions, rogue punctuation, and platform noise while preserving crucial semantic tokens and disaster keywords.
* **Dual-Stage Model Architecture:**
  * **Lightweight Baseline:** Fast TF-IDF vectorization paired with calibrated Logistic Regression for rapid CPU inference.
  * **Transformer Model:** Fine-tuned `distilbert-base-uncased` capturing deep bidirectional semantic relationships and context.
* **Automated Data Normalization (`notebooks/02b_prepare_kaggle_data.py`):** Automatically detects varied schema definitions (`text`, `tweet`, `target`, `label`) and handles multi-format ground truth mappings across massive datasets (~95k+ rows).
* **Interactive Web Application:** Built with Streamlit for single-text inference, live preprocessing inspection, and probability confidence visualization.

---

## 📐 System Architecture

```text
[ Raw Social Media Stream / Kaggle Dataset ]
                       │
                       ▼
      ┌─────────────────────────────────┐
      │  Modular Cleaner (preprocess.py) │
      │  - Lowercasing, URL/Mention Cut │
      │  - Hashtag Symbol Stripping     │
      │  - Whitespace Normalization     │
      └─────────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
┌──────────────────┐       ┌──────────────────────┐
│  TF-IDF Vectorizer│       │ DistilBERT Tokenizer │
│  (Max Feat: 1000)│       │ (Max Length: 128)    │
└──────────────────┘       └──────────────────────┘
         │                           │
         ▼                           ▼
┌──────────────────┐       ┌──────────────────────┐
│Logistic Regression│       │ Fine-Tuned DistilBERT│
│ (Fast Inference) │       │ (Contextual Nuance)  │
└──────────────────┘       └──────────────────────┘
         │                           │
         └─────────────┬─────────────┘
                       ▼
      ┌─────────────────────────────────┐
      │   Streamlit Web Interface       │
      │   - Real-Time Text Input        │
      │   - Class Label & Confidence    │
      └─────────────────────────────────┘
```

## 📂 Repository Structure

```
disaster-misinfo-detection/
├── data/
│   ├── raw/                       # Untouched raw datasets (train.csv, sample CSVs)
│   └── processed/                 # Standardized, cleaned data ready for training
├── models/
│   ├── logistic_model.pkl         # Serialized baseline classifier
│   ├── tfidf_vectorizer.pkl       # Fitted vectorizer vocabulary
│   └── distilbert_disaster/       # Fine-tuned PyTorch Transformer checkpoints
├── notebooks/
│   ├── 01_data_exploration.ipynb  # Initial EDA and sample dataset verification
│   ├── 02_baseline_model.ipynb    # Baseline ML model training & metrics
│   ├── 02b_prepare_kaggle_data.py # Dynamic ingestion pipeline for large datasets
│   └── 03_bert_model.ipynb        # Hugging Face DistilBERT fine-tuning pipeline
├── src/
│   └── preprocess.py              # Reusable NLP preprocessing functions
├── app/
│   └── main.py                    # Streamlit deployment application
├── .gitignore                     # Git ignore rules for data, models, and caches
├── requirements.txt               # Full dependency specification
└── README.md                      # Project documentation
```

## 🚀 Getting Started

### Prerequisites

Python 3.10 or higher

Recommended: Virtual environment (venv or conda)

Optional: NVIDIA GPU with CUDA drivers (for accelerating Transformer training)

### Installation

Clone the repository:

```bash
git clone [https://github.com/JkRoyal0311/disaster-misinfo-detection.git](https://github.com/JkRoyal0311/disaster-misinfo-detection.git)
cd disaster-misinfo-detection
```

Create and activate a virtual environment:

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 🧹 Data Pipeline & Preprocessing

The preprocessing module (src/preprocess.py) standardizes noisy disaster posts into clean representations:

```python
from src.preprocess import clean_text

raw_tweet = "🚨 URGENT: Flood water crossing safe mark in Sector 4! Evacuate now! [https://t.co/alert](https://t.co/alert) #AssamFloods @NDRFHQ"
clean = clean_text(raw_tweet)
print(clean)
# Output: "urgent flood water crossing safe mark in sector 4 evacuate now assamfloods"
```

To parse and clean large-scale raw data (e.g., Kaggle's 95,000+ row dataset placed at data/raw/train.csv):

```bash
cd notebooks
python 02b_prepare_kaggle_data.py
```

This will output data/processed/large_cleaned_disaster_news.csv with standardized columns: cleaned_text and binary label_encoded (0 = Real, 1 = Fake).

## 🧠 Model Training & Evaluation

### 1. Baseline Model (TF-IDF + Logistic Regression)

Located in notebooks/02_baseline_model.ipynb:

Transforms text using unigram/bigram TF-IDF features.

Fits a generalized linear model with L2 regularization.

Saves logistic_model.pkl and tfidf_vectorizer.pkl directly into models/.

### 2. Deep Learning Model (DistilBERT Fine-Tuning)

Located in notebooks/03_bert_model.ipynb:

Employs distilbert-base-uncased with sequence classification heads.

Implements dynamic token truncation and padding (max_length=128).

Employs AdamW optimizer, linear warmup schedules, and evaluation via Hugging Face Trainer.

| Model Architecture | Input Representation | Strengths | Use Case |
|---|---|---|---|
| Logistic Regression | TF-IDF (1,000 features) | Ultra-low latency, tiny footprint (~1MB) | High-throughput edge filtering |
| DistilBERT | WordPiece Embeddings | Understands contextual nuance & sarcasm | High-stakes verification |

## 💻 Running the Interactive Web App

Once models are generated in models/, launch the Streamlit frontend:

```bash
streamlit run app/main.py
```

The web dashboard will launch at http://localhost:8501.

Type or paste any news item, tweet, or WhatsApp forward.

Inspect the sanitized token stream generated by the preprocessor.

Receive an instant authenticity verdict accompanied by confidence scores.

## 🧪 Example Predictions

| Input Headline / Tweet | Cleaned Text | Ground Truth | Prediction |
|---|---|---|---|
| "NDRF teams deployed in Assam to assist flood-hit villagers." | ndrf teams deployed in assam to assist floodhit villagers | Real (0) | REAL NEWS (92.4%) |
| "NASA warns 9.8 earthquake hitting India tomorrow at 3 PM!!" | nasa warns 98 earthquake hitting india tomorrow at 3 pm | Fake (1) | FAKE NEWS (88.1%) |
| "Fake donation link circulating on WhatsApp for Kerala flood relief." | fake donation link circulating on whatsapp for kerala flood relief | Real (0) | REAL NEWS (79.6%) |

## 🗺 Roadmap & Future Improvements

[x] Week 1: Environment setup, modular directory layout, and baseline text cleaner.

[x] Week 2: TF-IDF feature extraction, Logistic Regression baseline, and persistence.

[x] Week 3: Hugging Face DistilBERT fine-tuning pipeline on 95k+ records.

[x] Week 4: Streamlit web application deployment with real-time inference.

[ ] Multimodal Analysis: Incorporate image verification (CLIP/ResNet) to detect recycled disaster imagery.

[ ] Multilingual Translation: Extend support for regional Indian languages (Hindi, Bengali, Odia, Assamese).

[ ] REST API: Expose inference endpoints via FastAPI with Docker containerization.

## 🤝 Contributing

Contributions, bug reports, and feature requests are welcome!

Fork the repository.

Create a descriptive feature branch: git checkout -b feature/AmazingFeature.

Commit your changes: git commit -m "Add AmazingFeature".

Push to the branch: git push origin feature/AmazingFeature.

Open a Pull Request.

## 📄 License

Distributed under the MIT License. See LICENSE for more information.

Developed with ❤️ to empower emergency response operations and mitigate digital disaster panic.
