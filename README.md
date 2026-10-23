# Multi-Level Toxicity Detection in Online Gaming Chats Using Machine Learning and NLP

## Project Overview

This project focuses on detecting and classifying toxic messages in online gaming chats using Machine Learning (ML) and Natural Language Processing (NLP). It aims to identify different levels of toxicity to help create safer and more respectful online gaming environments.

The system classifies chat messages into five categories:

- **Clean**
- **Mild**
- **Moderate**
- **Severe**
- **Extreme**

## Objectives

- Detect toxic language in online gaming chat messages.
- Classify messages according to their toxicity levels.
- Compare different machine learning and deep learning approaches.
- Apply text preprocessing and leetspeak normalization to improve text analysis.
- Develop a user interface using Streamlit for toxicity detection.

## Models Used

The project explores three approaches:

1. **Multinomial Naive Bayes** — a machine learning model used for text classification.
2. **Random Forest** — an ensemble learning model used to classify messages based on extracted text features.
3. **DistilBERT** — a transformer-based language model fine-tuned for toxicity classification.

## Technologies Used

- Python
- Natural Language Processing (NLP)
- Scikit-learn
- TF-IDF Vectorization
- DistilBERT
- Jupyter Notebook
- Streamlit
- Pickle (`.pkl`) for saving model-related objects

## Dataset

The project uses the **Jigsaw Toxic Comment dataset** for toxicity classification.

Text preprocessing includes cleaning and normalizing text, including leetspeak normalization, to help the models process gaming chat messages.

## Project Files

| File | Description |
|---|---|
| `app.py` | Streamlit application for the user interface and toxicity detection |
| `backend.ipynb` | Jupyter Notebook containing the project's backend code and experiments |
| `label_encoder.pkl` | Saved label encoder used for handling class labels |
| `Naive Bayes model.pkl` | Saved Naive Bayes model |
| `TF-IDF vectorizer.pkl` | Saved TF-IDF vectorizer for transforming text into numerical features |

**Note:** The `random_forest_model.pkl` file is not included in this repository because of its large file size (approximately 284 MiB). The Random Forest model may need to be regenerated from the notebook or obtained separately. The application may require modification or the missing model file if it loads the Random Forest model at startup.

## Reported Results

According to the project's evaluation:

- **Random Forest:** Reported weighted F1-score of 0.9112.
- **DistilBERT:** Reported accuracy of 92.20%, with a weighted F1-score of 0.6570.

These results illustrate why multiple evaluation metrics are useful when comparing models for multi-class toxicity classification.

## How to Run

1. Clone or download this repository.
2. Install the required Python packages.
3. Ensure that all model files required by the application are available.
4. Run the Streamlit application:

   ```bash
   streamlit run app.py
   ```

**Important:** The application may not run successfully if it requires the missing Random Forest model file. Refer to `backend.ipynb` to review the training workflow and regenerate the model if the notebook contains the necessary training code.

## Future Improvements

- Improve classification performance across all toxicity levels.
- Test the system on additional gaming chat datasets.
- Improve the user interface and provide clearer explanations of predictions.

## Project Purpose

This project demonstrates the application of machine learning and NLP techniques to toxicity detection, with a focus on identifying different levels of harmful language in online gaming conversations.
