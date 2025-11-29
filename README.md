# 🎬 IMDB Movie Review Sentiment Analysis with ML Benchmarking

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success)

This project is a comprehensive Machine Learning solution for **Sentiment Analysis** on the IMDB Movie Reviews dataset. It benchmarks four different classification algorithms to find the optimal model and includes a **multilingual inference pipeline** capable of translating non-English reviews for real-time sentiment prediction.

## 📌 Project Overview

The goal of this project is to classify movie reviews as either **Positive** or **Negative**. The pipeline includes advanced text preprocessing, TF-IDF vectorization, and a comparative analysis of multiple algorithms.

**Key Features:**
* **Text Preprocessing:** HTML tag removal, regex cleaning, stopword removal (NLTK).
* **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency) with top 5,000 features.
* **Model Benchmarking:** Comparison of Logistic Regression, Linear SVM, Multinomial Naive Bayes, and Random Forest.
* **Advanced Evaluation:** Confusion Matrix Heatmaps and ROC/AUC Curve visualizations.
* **Multilingual Support:** Integration with `deep-translator` to handle and classify reviews in languages other than English (e.g., Turkish, Spanish).

## 📂 Dataset

* **Source:** [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
* **Size:** 50,000 Reviews (Balanced: 25k Positive, 25k Negative)
* **Split:** 80% Training, 20% Testing

## 🛠️ Technologies Used

* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn
* **NLP:** NLTK
* **Visualization:** Matplotlib, Seaborn
* **Translation:** Deep-translator

## 📊 Model Performance Results

After training and evaluating four models, **Logistic Regression** achieved the highest accuracy, followed closely by Linear SVM.

| Model | Accuracy | F1 Score | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **88.99%** | **0.89** | 0.88 | 0.90 |
| Linear SVM | 88.29% | 0.88 | 0.87 | 0.89 |
| Multinomial NB | 85.57% | 0.85 | 0.85 | 0.86 |
| Random Forest | 85.03% | 0.84 | 0.85 | 0.83 |

*Note: Logistic Regression was selected as the production model due to its superior accuracy and inference speed.*

## 📈 Visualizations

The project generates the following plots to evaluate model performance:
1.  **Confusion Matrix Heatmap:** To visualize True Positives vs False Positives across all models.
    ![Confusion Matrix](conf_matrix_all.png)
2.  **ROC Curve:** Comparing the AUC (Area Under Curve) scores.
    ![ROC Curve](roc_auc.png)

## 🚀 Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/imdb-sentiment-analysis.git](https://github.com/yourusername/imdb-sentiment-analysis.git)
cd imdb-sentiment-analysis
