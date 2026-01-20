# Toxic Comment Detection

## 🧠 Problem Overview
Online platforms receive a massive volume of user-generated comments, some of which may contain toxic, abusive, or harmful language. Manual moderation is expensive, subjective, and does not scale.

This project builds a **machine learning system to automatically detect toxic comments**, with a particular focus on **reducing false positives**, so that non-toxic or sarcastic comments are not incorrectly removed.

The task is framed as a **binary text classification problem** using classical NLP techniques.

---

## 📊 Dataset
- Source: **Jigsaw Toxic Comment Classification Challenge (Kaggle)**
- Size: ~160,000 comments
- Original labels:
  - toxic
  - severe_toxic
  - obscene
  - threat
  - insult
  - identity_hate

The multi-label annotations were combined into a single binary label:
- `is_toxic = 1` if any toxicity label is present
- `is_toxic = 0` otherwise

⚠️ Dataset is **not included** due to licensing restrictions.

---

## 🛠️ Methodology

### 1. Data Cleaning
- Removed duplicate comments
- Removed empty or whitespace-only comments
- Inspected data types and label distributions

### 2. Baseline Model
- **Word-level TF-IDF**
- **Logistic Regression**
- Served as a reference point for all improvements

### 3. Error Analysis
- Analyzed false positives and false negatives
- Identified keyword bias and sarcasm-related misclassifications

### 4. Attempted Improvement (Rejected)
- Threshold tuning increased recall
- Precision collapsed, making the model unsuitable for moderation
- This approach was intentionally discarded

### 5. Final Model
- **Combined word-level and character-level TF-IDF features**
- Logistic Regression classifier
- Balanced precision and recall more effectively

---

## 📈 Results

Final evaluation on validation data:

| Class     | Precision | Recall   | F1-score |
|-----------|-----------|----------|---------|
| Non-toxic | 0.96      | 0.99     | 0.98 |
| Toxic     | **0.91**  | **0.66** | **0.77** |

---

## 🔍 Key Insights
- Keyword-based models lead to high false positives
- Character n-grams improve robustness to spelling and obfuscation
- Error analysis guided more meaningful improvements than blind tuning
- Classical ML methods remain strong for structured NLP tasks

---

## 🧰 Tech Stack
- Python
- Pandas, NumPy
- scikit-learn
- TF-IDF
- Logistic Regression

---

## 💡 Future Work
- Precision–recall threshold tuning for deployment
- Transformer-based models (e.g., BERT)
- Multi-label toxicity classification
