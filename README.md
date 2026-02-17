
# Fake News Detection using Machine Learning

## Project Overview

This project focuses on detecting whether a news article is **Fake** or **Real** using multiple Machine Learning algorithms.

The system performs:

* Text preprocessing
* Feature extraction using TF-IDF
* Training using multiple classifiers
* Manual testing for real-time prediction

The goal is to compare different ML models and evaluate their performance on fake news classification.

---

## Dataset Used

The dataset consists of two CSV files:

* `Fake.csv` → Contains fake news articles
* `True.csv` → Contains real news articles

Each dataset includes:

* Title
* Text
* Subject
* Date

The datasets are merged and labeled:

* `0` → Fake News
* `1` → Real News

---

## Technologies & Libraries Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Regex (Text Cleaning)
* TF-IDF Vectorizer

---

## Project Workflow

### Data Preprocessing

* Added target column (`class`)
* Removed unnecessary columns (`title`, `subject`, `date`)
* Cleaned text using:

  * Lowercasing
  * Removing punctuation
  * Removing URLs
  * Removing HTML tags
  * Removing numbers
  * Removing special characters

Custom cleaning function:

```python
def wordopt(text):
```

---

### 2️Train-Test Split

* 75% Training data
* 25% Testing data

```python
train_test_split(test_size=0.25)
```

---

### 3️Feature Extraction

Text data converted into numerical format using:

```python
TfidfVectorizer()
```

TF-IDF helps in:

* Identifying important words
* Reducing weight of common words
* Converting text to feature vectors

---

## Machine Learning Models Used

The project compares four classifiers:

### 1. Logistic Regression

* Good baseline classifier
* Works well for text classification

### 2. Decision Tree

* Tree-based model
* Easy to interpret

### 3. Gradient Boosting Classifier

* Boosting technique
* Combines weak learners

### 4. Random Forest Classifier

* Ensemble learning
* Reduces overfitting
* More stable predictions

Each model is evaluated using:

* Accuracy Score
* Classification Report (Precision, Recall, F1-Score)

---

## Manual Testing Feature

The project includes a function for real-time prediction:

```python
manual_testing(news)
```

User inputs a news statement, and the system predicts:

* Fake News
* Not A Fake News

Predictions are generated from all four models for comparison.

---

## Evaluation Metrics

* Accuracy Score
* Precision
* Recall
* F1-Score

These metrics help determine how well each model performs.

---

## How to Run the Project

1. Clone the repository:

```
git clone https://github.com/your-username/fake-news-detection.git
```

2. Install dependencies:

```
pip install pandas numpy scikit-learn matplotlib seaborn
```

3. Place dataset files inside the project directory.

4. Run:

```
python fake_news_detection.py
```

5. Enter news text when prompted for manual testing.

---

## Key Features

✅ Text Cleaning Pipeline
✅ TF-IDF Feature Extraction
✅ Multiple ML Model Comparison
✅ Manual User Testing
✅ Performance Evaluation

---

## Future Improvements

* Add Deep Learning (LSTM / BERT)
* Deploy using Flask or Streamlit
* Create Web Interface
* Hyperparameter Tuning
* Model Saving using Pickle


