# Rate Beer – Machine Learning Classification Project
Student: Pedro Azevedo (47094)
Course: Machine Learning (LEIM)
Date: January 21, 2024

## Project Overview
This project implements machine learning classification techniques to predict beer quality based on textual reviews. The dataset consists of beer reviews with individual component ratings (smell, taste, look, feel) and an overall quality score, alongside detailed text reviews.

Objective: Determine beer quality based on written review text using natural language processing and classification algorithms.

## Dataset Description
Data Characteristics

Rating Components:

 - Smell, Taste, Look, Feel: Scale of 1–5
 - Overall Quality: Scale of 1–10

Data Statistics (Training Set):

 - Overall Mean: 5.52 | Median: 6 | Mode: 6
 - Taste Mean: 3.03 | Median: 3 | Mode: 3
 - Smell Mean: 3.2 | Median: 3 | Mode: 3
 - Look Mean: 3.63 | Median: 4 | Mode: 4
 - Feel Mean: 3.48 | Median: 3 | Mode: 3

## Methodology
### Text Preprocessing & Vocabulary Construction
Data Cleaning:

 - Remove HTML tags (<br>)
 - Extract only alphabetic characters using regular expressions
 - Apply stemming algorithms to reduce words to root forms

Stemming Algorithms Tested:

 - PorterStemmer
 - SnowballStemmer
 - LancasterStemmer

Feature Vectorization (TF-IDF):

 - Technique: Term Frequency-Inverse Document Frequency (TF-IDF) matrix

Parameters tuned:

 - min_df: Minimum document frequency
 - token_pattern: Regular expression for word extraction
 - stop_words: Remove common English words
 - n-grams: Unigrams, bigrams, trigrams

### Classification Approaches

Binary Classification:

Approach 1: Multi-class reduction

 - Very Good (overall ≥ 9) vs. all others
 - Very Bad (overall ≤ 2) vs. all others

Approach 2: Balanced classification

 - Very Good (≥ 9) vs. Very Bad (≤ 2)
 - Ignore intermediate reviews

Multi-class Classification:

 - Predict three components: Smell, Taste, Overall
 - Uses Logistic Regression with hyperparameter tuning

### Classifiers Implemented

 - Logistic Regression: Weight-based feature importance
 - Support Vector Machines (SVM): Margin-based decision boundary
 - Truncated SVD: Dimensionality reduction for sparse matrices

## Results Summary
Binary Classification (Approach 2: Balanced)
|Classifier |	Errors |	Score |	Accuracy|	Precision	|Recall|	F1-Score|
|-----------|----------|----------|---------|---------------|------|------------|
|Logistic Regression|	215|	0.9393|	94%	|0.9473	|0.9635|	0.9553|
|SVM	|205	|0.9421	|94%	|0.9510|	0.9641|	0.9575|

→ Both classifiers perform excellently on balanced data; SVM slightly outperforms Logistic Regression.

Multi-class Classification Results

| Component | Errors | Score | Accuracy |
|-----------|--------|-------|----------|
| Smell | 13,567 | 0.4573 | 45.74% |
| Taste | 13,253 | 0.4699 | 46.92% |
| Overall | 17,435 | 0.3026 | 30.30% |

→ Poor performance; high error rates on neutral/intermediate classes.

## Key Findings
Class Imbalance Crisis: Approach 1 (including all reviews) suffers from severe class imbalance, causing overfitting and near-zero recall for minority classes.

Balanced Data Wins: Approach 2 achieves ~94% accuracy, proving balanced training data is critical.

SVM > Logistic Regression: SVM marginally outperforms (0.9421 vs. 0.9393 score) with superior ROC metrics.

Review Ambiguity: Descriptive, non-sentiment reviews or reviews with mixed sentiment are frequently misclassified.

Neutral Bias: Predicting intermediate ratings (3–4) is significantly harder due to subtle linguistic patterns.

Stemming Matters: PorterStemmer and LancasterStemmer excel for binary tasks; SnowballStemmer for multi-class.

### Recommendations
Implement sentiment analysis preprocessing

Explore deep learning (LSTM, Transformers) for context understanding

Apply class weighting for multi-class imbalance

Use pre-trained embeddings (Word2Vec, BERT) instead of TF-IDF

Test ensemble methods (Gradient Boosting, Random Forest)

Perform error analysis on edge cases

## Technologies Used
Language: Python 3.x

Libraries:

scikit-learn (classification, TF-IDF)

numpy, pandas (data handling)

nltk (text preprocessing)

matplotlib, seaborn (visualization)

Algorithms: Logistic Regression, SVM, Truncated SVD
