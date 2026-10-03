# SpamShield: SMS & Email Classifier

A machine learning project that classifies text messages as "ham" (legitimate) or "spam" using Natural Language Processing (NLP).

## 🚀 Overview

SpamShield uses a trained `StackingClassifier` to analyze incoming messages and detect unsolicited or malicious content in real-time.

## 🛠 Features

- **Robust Preprocessing**: Uses NLTK for tokenization, stopword removal, and stemming with the Porter Stemmer.
- **Ensemble Intelligence**: A `StackingClassifier` (SVC + Naive Bayes + ExtraTrees) meta-model, providing high precision and reliability.
- **Streamlit Interface**: Clean, dark-mode focused UI that displays the classification verdict and message statistics (char/word count, processed tokens).

## 📊 Performance

| Metric | Score |
| :--- | :--- |
| **Accuracy** | ~97.7% |
| **Precision** | ~92.5% |

## 💻 Tech Stack

- **ML Frameworks**: Scikit-Learn, XGBoost
- **NLP**: NLTK (Natural Language Toolkit)
- **Frontend/API**: Streamlit
- **Visualization**: Seaborn, Matplotlib

## ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/rajendra-pal/sms-email-classifier.git
   cd sms-spam-classifier
   ```

2. **Setup virtual environment (recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

Run the web interface:

```bash
streamlit run app.py
```

Paste an SMS or email message in the text area and click **Analyze**.


