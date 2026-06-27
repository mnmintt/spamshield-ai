# SpamShield AI — Email & SMS Spam Detector

An intelligent spam detection web application built with NLP and Machine Learning. Detects spam in typed messages, uploaded files (PDF, DOCX, TXT), images, and bulk CSV uploads using both traditional ML models and advanced transformer-based NLP (DistilBERT).

## Team Members

| Name | Role |
|------|------|
| Muhammad Ilham | Data & NLP Pipeline |
| Chan Zi Chao | Model Training & Evaluation |
| Muhammad Iskandar Zulkarnain | Streamlit Application |
| Tey Yu Yang | Visualizations & Documentation |

## How to Run Locally

**1. Clone the repository**
```
git clone https://github.com/mnmintt/spamshield-ai.git
cd spamshield-ai
```

**2. Install dependencies**
```
pip install -r requirements.txt
```

**3. Run the app**
```
streamlit run app.py
```

**4. Open in browser**
```
http://localhost:8501
```

## Features

| Feature | Description |
|---------|-------------|
| Text Analyzer | Paste any message and get instant spam prediction with confidence score and trigger word highlights. Supports SVM + TF-IDF and DistilBERT Transformer models |
| File Scanner | Upload .txt, .pdf, or .docx files and scan their full content for spam |
| Image Scanner | Upload an image (PNG, JPG) containing text and scan it for spam using OCR (EasyOCR) |
| Batch Prediction | Upload a CSV with multiple messages and analyze them all at once with downloadable results |
| Data Explorer | Browse and search the full training dataset with filters |
| Visualizations | Word clouds, top word frequency, confusion matrix, and model comparison charts |
| Model Info | Full breakdown of NLP pipeline, model metrics, feature extraction methods, and DistilBERT details |
| History | View all messages analyzed in the session with filters, model info, and CSV export |

## Project Structure

```
spamshield-ai/
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── data/
│   ├── spam.csv                  # Original dataset
│   ├── spam_cleaned.csv          # Preprocessed dataset
│   ├── labels.csv                # Encoded labels
│   ├── confusion_matrix.png      # SVM model confusion matrix
│   ├── bert_confusion_matrix.png # DistilBERT confusion matrix
│   ├── model_comparison_chart.png
│   └── model_comparison.csv      # All 4 model results
├── models/
│   ├── best_model.pkl            # Deployed SVM + TF-IDF model
│   └── tfidf_vectorizer.pkl      # Fitted TF-IDF vectorizer
└── notebooks/
    └── NLP_Project.ipynb         # Full pipeline notebook
```

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes (TF-IDF) | 96.7% | 99.1% | 75.8% | 85.9% |
| SVM (TF-IDF) | 98.3% | 99.2% | 87.9% | 93.2% |
| SVM (Word2Vec) | 94.5% | 87.3% | 69.1% | 77.2% |
| **DistilBERT Transformer** | **99.6%** | **100%** | **97.3%** | **98.6%** |

DistilBERT achieved the highest F1-Score of 98.6%, outperforming all traditional ML models.

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Web App | Streamlit |
| ML Models | Scikit-learn (LinearSVC, MultinomialNB) |
| Advanced NLP | HuggingFace Transformers (DistilBERT) |
| NLP Preprocessing | NLTK, Gensim (Word2Vec) |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, WordCloud |
| File Reading | PyPDF2, python-docx |
| Image OCR | EasyOCR |

## Dataset

- **Name:** SMS Spam Collection Dataset
- **Source:** Kaggle
- **Size:** 5,572 messages (4,825 ham, 747 spam)
- **Features:** Raw text message + label (spam/ham)

## Advanced NLP

This project implements **DistilBERT** (`mariagrandury/distilbert-base-uncased-finetuned-sms-spam-detection`) as an advanced NLP model alongside traditional ML models. DistilBERT is a lightweight transformer model that:

- Retains 97% of BERT's language understanding capability
- Is 40% smaller and 60% faster than full BERT
- Was fine-tuned specifically on SMS spam detection data
- Achieved 99.6% accuracy and 98.6% F1-Score on our test set

## Live Demo

Deployed on Streamlit Cloud: https://spamshield-ai-umseqzr7myjbprkdapyvgd.streamlit.app
