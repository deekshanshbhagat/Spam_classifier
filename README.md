# Spam Detection using Machine Learning

A machine learning-based spam classifier that predicts whether a given message is **spam** or **ham**. It uses Natural Language Processing (NLP) techniques for text preprocessing, vectorization, and classification. The best-performing model is deployed with a Streamlit UI for real-time predictions.

## Video demo
[Demo:](https://drive.google.com/file/d/1o12bfrmcbyGsWZFFcWWN8BZi_ly-T3c5/view?usp=sharing)

---

## Project Workflow

### 1. Dataset Preparation
- Used the **SMS Spam Collection Dataset**, which contains labeled SMS messages as `spam` or `ham`.
- Loaded and cleaned the dataset using `pandas`.

### 2. Exploratory Data Analysis (EDA)
- Visualized class distribution of spam vs ham messages.
- Analyzed message lengths, most frequent words, and text patterns.
- Used word clouds and bar plots for insights.

### 3. Text Preprocessing and Vectorization
- Applied NLP techniques such as:
  - Lowercasing
  - Removing punctuation, special characters, stopwords
  - Tokenization and stemming
- Converted text into numerical features using **TF-IDF Vectorizer**.

### 4. Model Building and Training
- Tried and compared several classifiers:
  - Multinomial Naive Bayes ✅ *(Best Performance)*
  - Support Vector Classifier (SVC)
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Logistic Regression
  - Random Forest
  - AdaBoost
  - Bagging
  - Extra Trees
  - Gradient Boosting
  - XGBoost

### 5. Model Evaluation
- Evaluated models using metrics such as:
  - Accuracy : 0.975
  - Precision : 0.974
- Naive Bayes was selected based on high precision.

---

##  Streamlit Web App

The project includes a Streamlit app (`app.py`) for interactive use:

- Users can input any message
- The model will predict whether it's spam or ham
- Uses saved `model.pkl` and `vectorizer.pkl`

###  How to Run

```bash
git clone https://github.com/deekshanshbhagat/Spam_classifier.git
cd Spam_classifier
pip install -r requirements.txt
streamlit run app.py

