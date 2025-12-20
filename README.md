# Nepali_classification

# Introduction

This project focuses on classifying Nepali news articles into predefined categories using Natural Language Processing (NLP) techniques. Two different approaches—a traditional machine learning model and a transformer-based deep learning model—are implemented and compared to understand their effectiveness on Nepali-language text.

# Dataset and Problem Statement

A publicly available Nepali news classification dataset from Hugging Face was used. The dataset contains approximately 2,000 Nepali news articles labeled into multiple news categories. The task is a multi-class text classification problem, where the goal is to predict the correct category for a given Nepali news article.

# Model 1: TF-IDF + Logistic Regression

The first model uses TF-IDF (Term Frequency–Inverse Document Frequency) to convert Nepali text into numerical feature vectors. Logistic Regression is then applied as a classifier. This approach is lightweight, fast to train, and provides high interpretability through feature importance analysis.

# Model 2: Multilingual BERT (mBERT)

The second model is based on a pre-trained transformer, bert-base-multilingual-cased. The model is fine-tuned on the Nepali news dataset by adding a classification head. This approach captures contextual and semantic information in Nepali text more effectively than traditional methods.

# Training Information

For the TF-IDF model, training involves fitting the vectorizer and classifier on the training split. For mBERT, the model is fine-tuned for multiple epochs using the Hugging Face Trainer API with GPU acceleration. Proper tokenization, padding, and truncation are applied during training.

# Evaluation Metrics

Both models are evaluated using standard classification metrics including accuracy, precision, recall, and F1-score. Confusion matrices are used to analyze class-wise performance and misclassification patterns.

# Model Performance

The TF-IDF + Logistic Regression model achieved an accuracy of approximately 78%, demonstrating strong baseline performance with minimal computational cost. The mBERT model achieved a higher accuracy of approximately 84%, indicating improved understanding of contextual and semantic relationships in Nepali text.

# Visualization and Model Interpretation

For the TF-IDF model, feature importance analysis highlights the most influential words contributing to predictions. For the mBERT model, attention visualization and gradient-based saliency maps are used to understand which parts of the input text influence the model’s decisions. Training loss curves are also analyzed to monitor convergence.

# Comparison of Models

The TF-IDF model is faster, simpler, and more interpretable, making it suitable for resource-constrained environments. In contrast, mBERT provides better accuracy and contextual understanding at the cost of higher computational requirements. This comparison highlights the trade-off between efficiency and performance.

# Additional Observations

The experiment demonstrates the importance of dataset size and model choice in NLP tasks. Pre-trained transformer models require fine-tuning for downstream tasks, while traditional models remain competitive on smaller datasets.

# gConclusion

This project successfully demonstrates Nepali news classification using both traditional and transformer-based approaches. The results show that while mBERT achieves superior accuracy, TF-IDF with Logistic Regression remains a strong and interpretable baseline. The study highlights practical considerations in choosing models for low-resource language NLP tasks.🎯 Align it exactly with your assignment rubric wording

