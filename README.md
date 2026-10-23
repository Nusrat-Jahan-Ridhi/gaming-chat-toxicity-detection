# 🎮 Multi-Level Toxicity Detection in Online Gaming Chats Using Machine Learning and NLP

## 📌 Project Overview

Online gaming communities allow players to communicate and collaborate, but they can also expose users to cyberbullying, hate speech, and aggressive language. Traditional keyword-based filters can be bypassed using gaming slang and leetspeak, such as `k1ll` or `h@te`.

This project develops a **multi-level toxicity detection system** using Machine Learning (ML) and Natural Language Processing (NLP) to classify online gaming chat messages into five toxicity severity levels. It combines traditional machine learning models with a transformer-based model and includes a Streamlit-based user interface for toxicity detection.

## 🎯 Project Objectives

- Detect toxic language in online gaming chats.
- Classify messages into five levels of toxicity rather than using only binary toxic/non-toxic classification.
- Apply text preprocessing and leetspeak normalization to handle noisy text and intentional misspellings.
- Compare traditional machine learning models with a transformer-based deep learning model.
- Explore a practical approach to identifying toxic communication in online gaming environments.

## 🤖 Models Implemented

The project compares three classification models:

| Model | Approach |
|---|---|
| Multinomial Naive Bayes | Probabilistic machine learning |
| Random Forest | Ensemble machine learning |
| DistilBERT | Transformer-based deep learning |

## 🏷️ Toxicity Classification

Messages are categorized into five severity levels using rule-based mapping of the original Jigsaw dataset labels.

| Toxicity Level | Original Dataset Labels |
|---|---|
| Clean | None |
| Mild | Toxic, Insult |
| Moderate | Obscene |
| Severe | Severe Toxic, Identity Hate |
| Extreme | Threat, Threat Keywords |

## 🛠️ Technologies and Techniques

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Natural Language Processing:** NLTK and text preprocessing
- **Feature Extraction:** TF-IDF vectorization with unigram and bigram features
- **Deep Learning:** DistilBERT and Hugging Face Transformers
- **Application Interface:** Streamlit
- **Development Environment:** Jupyter Notebook

## 🧹 Methodology

The project follows these main stages:

1. **Dataset loading:** Use the Jigsaw Toxic Comment Classification dataset.
2. **Rule-based categorization:** Map the original toxicity labels into five severity levels.
3. **Text preprocessing:** Convert text to lowercase and remove HTML tags, URLs, punctuation, numbers, and extra whitespace.
4. **Leetspeak normalization:** Normalize substitutions such as `h4te` to `hate`.
5. **Feature extraction:** Use TF-IDF with unigram and bigram features for traditional machine learning models.
6. **Model training:** Train Multinomial Naive Bayes and Random Forest, and fine-tune DistilBERT using tokenized text.
7. **Evaluation:** Compare model performance using accuracy, weighted precision, weighted recall, and weighted F1-score.
8. **User interface:** Provide a Streamlit-based interface for toxicity detection.

The report describes an **80:20 stratified training/testing split**. DistilBERT was fine-tuned with a maximum sequence length of 128 and two training epochs.

## 📊 Model Performance

The following results are reported in the project paper.

| Evaluation Metric | Multinomial Naive Bayes | Random Forest | DistilBERT |
|---|---:|---:|---:|
| Test Accuracy | 89.59% | **92.39%** | 92.20% |
| Weighted Precision | 0.8748 | **0.9092** | 0.7020 |
| Weighted Recall | 0.8959 | **0.9237** | 0.6270 |
| Weighted F1-Score | 0.8617 | **0.9112** | 0.6570 |

### 🔍 Key Findings

- **Random Forest achieved the best overall balance** across the reported evaluation metrics.
- Multinomial Naive Bayes provided a useful baseline for toxicity classification.
- DistilBERT achieved competitive test accuracy but a substantially lower weighted F1-score, indicating weaker overall performance across the toxicity classes.
- The report discusses short text sequences and class imbalance as possible challenges for transformer-based classification in gaming chat scenarios.

## 📁 Repository Structure

The repository contains the following project files:

```text
gaming-chat-toxicity-detection/
├── app.py
├── backend.ipynb
├── label_encoder.pkl
├── Naive Bayes model.pkl
└── TF-IDF vectorizer.pkl
```

**Note:** `random_forest_model.pkl` is not included in the repository because of its large file size (approximately 284 MiB). The report's Random Forest results are included above, but the saved model file is not currently available in the repository. If the application requires this file, it must be provided separately or regenerated using the appropriate training code.

## 🚀 Running the Application

### Prerequisites

- Python installed on your system
- The required Python libraries
- All model and preprocessing files required by the application

### Steps

1. Clone or download this repository.
2. Install the dependencies required by the project.
3. Ensure that the required model files are available.
4. Start the Streamlit application:

```bash
streamlit run app.py
```

**Important:** The application may not run as expected until the missing Random Forest model dependency is resolved. Check `app.py` and `backend.ipynb` for the exact model-loading and training requirements.

## 🔮 Future Improvements

As discussed in the report, possible future improvements include:

- Extending toxicity detection to multilingual content, particularly Bengali gaming chats.
- Integrating speech-to-text for voice-chat moderation.
- Connecting the system with gaming platforms and Discord servers.
- Evaluating larger, more diverse gaming-specific datasets.

## 📚 Dataset Reference

**Jigsaw Toxic Comment Classification Challenge — Kaggle**

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

## 👥 Academic Project

Developed as a machine learning laboratory project at the **Bangladesh University of Professionals (BUP)**.

The project explores how Machine Learning and NLP can help identify different levels of toxic communication and contribute to safer online gaming communities.

