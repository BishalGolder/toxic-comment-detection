# Toxic Comment Detection

This project builds a machine learning model to detect toxic online comments.
The focus is on reducing false positives to avoid incorrect moderation.

## Dataset
- Source: Kaggle – Jigsaw Toxic Comment Classification Challenge
- Size: ~160k comments
- Labels: toxic, severe_toxic, obscene, threat, insult, identity_hate
- Converted to binary classification (`is_toxic`)

## Method
- Text preprocessing using Pandas
- Baseline: Word-level TF-IDF + Logistic Regression
- Error analysis on misclassified samples
- Attempted threshold tuning (rejected)
- Final model: Word + Character TF-IDF

## Result
Final toxic-class F1-score: **0.77**  
Precision: **0.91**, Recall: **0.66**

## Key Insight
Feature engineering worked more than aggressive threshold tuning.