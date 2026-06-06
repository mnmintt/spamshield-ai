# 🛡️ SpamShield AI — Email & SMS Spam Detector

An intelligent spam detection web application built with NLP and Machine Learning. Detects spam in typed messages, uploaded files (PDF, DOCX, TXT), and bulk CSV uploads.

## 👥 Team Members
- Member 1 — Data & NLP Pipeline
- Member 2 — Model Training & Evaluation
- Member 3 — Streamlit Application
- Member 4 — Visualizations & Documentation

## 🚀 How to Run Locally

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

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 Text Analyzer | Paste any message and get instant spam prediction with confidence score and keyword highlights |
| 📄 File Scanner | Upload .txt, .pdf, or .docx files and scan their content for spam |
| 📂 Batch Prediction | Upload a CSV with multiple messages and analyze them all at once with downloadable results |
| 📄 File Scanner | Upload .txt, .pdf, or .docx files and scan their content for spam |
| 📊 Data Explorer | Browse and search the full training dataset with filters |
| 📈 Visualizations | Word clouds, top word frequency, confusion matrix, and model comparison charts |
| 🤖 Model Info | Full breakdown of NLP pipeline, model metrics, and feature extraction methods |
| 🕓 History | View all analyzed messages in the session with filters and CSV export |

## 📁 Project Structure

```
spamshield-ai/
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── data/
│   ├── spam.csv                  # Original dataset
│   ├── spam_cleaned.csv          # Preprocessed dataset
│   ├── labels.csv                # Encoded labels
│   ├── confusion_matrix.png      # Model evaluation chart
│   ├── model_comparison_chart.png
│   └── model_comparison.csv
├── models/
│   ├── best_model.pkl            # Deployed SVM model
│   └── tfidf_vectorizer.pkl      # Fitted TF-IDF vectorizer
└── notebooks/
    └── NLP_Project.ipynb         # Full pipeline notebook
```

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes (TF-IDF) | 96.7% | 99.1% | 75.8% | 85.9% |
| **SVM (TF-IDF) ⭐ Best** | **98.3%** | **99.2%** | **87.9%** | **93.2%** |
| SVM (Word2Vec) | 94.6% | 96.3% | 69.1% | 77.4% |

## 🛠️ Tech Stack

| Category | Libraries |
|----------|-----------|
| Web App | Streamlit |
| ML Models | Scikit-learn (LinearSVC, MultinomialNB) |
| NLP | NLTK, Gensim (Word2Vec) |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, WordCloud |
| File Reading | PyPDF2, python-docx, EasyOCR |

## 📦 Dataset

- **Name:** SMS Spam Collection Dataset
- **Source:** Kaggle
- **Size:** 5,572 messages (4,825 ham, 747 spam)
- **Features:** Raw text message + label (spam/ham)

## 🌐 Live Demo

Deployed on Streamlit Cloud: (https://spamshield-ai-umseqzr7myjbprkdapyvgd.streamlit.app/)
