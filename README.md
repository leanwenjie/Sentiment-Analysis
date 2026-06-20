# Brain: Sentiment & Emotion Intelligence Dashboard

An interactive, premium-designed Streamlit dashboard powered by a dual-classification machine learning pipeline. It analyzes both **Sentiment** (Positive/Negative/Neutral) and **Emotion Category** (13 distinct emotions) using TF-IDF text features and Logistic Regression models.

This project implements robust, mathematical solutions to handle **severe class imbalance** and **noisy training labels** (e.g., general negative sentiment and profanities incorrectly labeled as neutral in the raw dataset).

---

## 🚀 Key Features

*   **🔍 Single Sentence Analyzer**: Real-time evaluation of custom sentences showing confidence gauge metrics and horizontal probability distributions using Plotly.
*   **📊 Dataset Diagnostics**: Side-by-side interactive bar charts illustrating the dataset distribution before and after class balancing.
*   **📈 Model Performance**: Detailed classification reports displaying Precision, Recall, and F1-Scores for all classes (validated on holdout testing sets).

---

## 🛠️ The Machine Learning Approach

### 1. Handling Class Imbalance
The original dataset has severe imbalance: **80.3%** of sentences are `neutral`. If trained directly, the model will heavily favor the neutral class.
*   **Sentiment Classifier**: Classes are strictly balanced to **55,000** samples per class before training.
*   **Emotion Classifier**: Classes are strictly balanced to **20,000** samples per class (majority class is downsampled, minority classes are over-sampled with replacement).

### 2. Label Noise Cleaning (Weak Supervision)
The raw `neutral` class contains a massive amount of general sentiment noise (e.g., words like *"terrible"*, *"fuck"*, *"awful"* were labeled as `neutral`). 
*   Before vectorization, we run an optimized **VADER-based Weak Supervision scanner** over the balanced neutral subset.
*   If VADER identifies a neutral sentence as strongly negative (score $\le -0.3$), it is programmatically corrected to `negative`/`sadness` before training.
*   If positive (score $\ge 0.3$), it is corrected to `positive`/`happiness`.

### 3. Preventing Baseline Bias (`fit_intercept=False`)
Because high-entropy classes (like `neutral`) contain highly diverse or featureless documents, the model naturally learns a massive positive intercept (bias term). This intercept dominates during inference, pulling short sentences from other classes into `neutral`.
*   We disable bias learning by setting `fit_intercept=False`. This forces classification decisions to be **100% feature-driven** without hardcoding rules.

---

## 📦 Prerequisites

Install the required Python packages:
```bash
pip install streamlit pandas numpy scikit-learn joblib plotly nltk
```

---

## 💻 How to Run

### Step 1: Train the Models
To process the dataset, apply label cleaning, balance classes, and save the pipelines:
```bash
python train.py
```
This saves the trained pipelines and metadata to the `models/` directory.

### Step 2: Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
├── models/
│   ├── sentiment_pipeline.joblib  # Trained Sentiment Model
│   ├── emotion_pipeline.joblib    # Trained Emotion Model
│   └── training_metadata.json     # Saved metrics and dataset stats
├── app.py                         # Streamlit Dashboard UI
├── train.py                       # Preprocessing and Training Pipeline
├── .gitignore                     # Configured to ignore the heavy CSV dataset
└── README.md                      # Documentation
```

---

## 🔗 Repository
[https://github.com/leanwenjie/Sentiment-Analysis.git](https://github.com/leanwenjie/Sentiment-Analysis.git)
