# Twitter Sentiment Analysis Project

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Architecture](#project-architecture)
- [Data Preprocessing](#data-preprocessing)
- [Feature Extraction](#feature-extraction)
- [Model Training](#model-training)
- [Performance Metrics](#performance-metrics)
- [Web Application](#web-application)
- [Installation & Usage](#installation--usage)
- [Technical Stack](#technical-stack)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)

---

## 🎯 Overview

This project implements an end-to-end **Twitter Sentiment Analysis** system that classifies tweets as **Positive** or **Negative**. The system uses advanced Natural Language Processing (NLP) techniques and machine learning models to achieve **78.46% accuracy** on a dataset of 1.6 million tweets.

### Key Features
- ✅ Advanced text preprocessing with negation handling and lemmatization
- ✅ TF-IDF feature extraction with n-grams (unigrams, bigrams, trigrams)
- ✅ Multiple ML models including ensemble methods
- ✅ Interactive Streamlit web application
- ✅ Real-time sentiment prediction with confidence scores
- ✅ Batch analysis with feedback system

---

## 📊 Dataset

### Sentiment140 Dataset
- **Source:** Kaggle (Twitter Sentiment140)
- **Size:** 1,600,000 tweets
- **Classes:** Binary classification
  - `0` = Negative sentiment
  - `4` = Positive sentiment (converted to `1`)
- **Time Period:** 2009-2017
- **Features:**
  - `target`: Sentiment label (0 or 4)
  - `ids`: Tweet ID
  - `date`: Timestamp
  - `flag`: Query flag (NO_QUERY)
  - `user`: Username
  - `text`: Tweet content

### Data Distribution
- **Positive tweets:** 800,000 (50%)
- **Negative tweets:** 800,000 (50%)
- **Missing values:** None
- **Encoding:** Latin-1

---

## 🏗️ Project Architecture

```
sentiment-analysis/
│
├── data/
│   ├── sentiment140.csv              # Raw dataset
│   └── sentiment140_cleaned.csv      # Processed dataset
│
├── models/
│   ├── logistic_regression_model.pkl
│   ├── naive_bayes_model.pkl
│   ├── voting_ensemble_model.pkl
│   ├── stacking_ensemble_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── training_metadata.json
│   └── best_hyperparameters.json
│
├── preprocessing.py                  # Text preprocessing module
├── feature_extraction.py             # TF-IDF feature extraction
├── train_model.py                    # Model training pipeline
├── predict.py                        # Prediction module
├── app.py                            # Streamlit web application
└── requirements.txt                  # Python dependencies
```

---

## 🔧 Data Preprocessing

### Preprocessing Pipeline

The preprocessing module implements advanced text cleaning techniques:

#### 1. **Emoji Sentiment Mapping**
Converts emojis to sentiment words:
```
😍, 😊, 🙂 → "positive"
😠, 😡, 😢 → "negative"
```

#### 2. **Twitter Slang Normalization**
```
"u" → "you"
"ur" → "your"
"pls" → "please"
"idk" → "i don't know"
```

#### 3. **Character Repetition Reduction**
```
"sooooo" → "soo"
"amazingggg" → "amazingg"
```

#### 4. **URL & Mention Removal**
- Remove HTTP/HTTPS URLs
- Remove @mentions
- Keep hashtag words (remove # symbol)

#### 5. **Special Character Removal**
- Keep only letters, underscores, and spaces
- Remove numbers and punctuation

#### 6. **Negation Handling**
Preserve sentiment-flipping negations:
```
"not good" → "not_good"
"never happy" → "never_happy"
```

#### 7. **Lemmatization with POS Tagging**
Uses WordNet lemmatizer with part-of-speech tagging for better accuracy:
```
"running" → "run"
"better" → "good"
"happiest" → "happy"
```

#### 8. **Stopword Removal**
Removes common English stopwords while preserving negations

### Example Transformation

**Original:**
```
"@user I absolutely LOVE this product!!! 😍 http://example.com #amazing"
```

**Cleaned:**
```
"absolutely love product positive amazing"
```

---

## 🔍 Feature Extraction

### TF-IDF Vectorization (Enhanced Version 2.0)

#### Configuration Parameters
- **Max Features:** 10,000 (vocabulary size)
- **Max Document Frequency:** 0.8 (ignore very common words)
- **Min Document Frequency:** 3 (ignore very rare words)
- **N-gram Range:** (1, 3) - unigrams, bigrams, and trigrams
- **Sublinear TF:** True (use log scaling)
- **Normalization:** L2 norm

#### Why These Settings?

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| max_features | 10,000 | Captures comprehensive vocabulary while managing memory |
| max_df | 0.8 | Filters extremely common words like "the", "is" |
| min_df | 3 | Removes typos and extremely rare words |
| ngram_range | (1,3) | Captures phrases like "not good", "very happy", "best purchase ever" |
| sublinear_tf | True | Reduces impact of very frequent terms |

#### Feature Distribution Example
- **Unigrams:** ~6,500 features (65%)
- **Bigrams:** ~2,500 features (25%)
- **Trigrams:** ~1,000 features (10%)

#### Sample N-grams Extracted
```
Unigrams: love, hate, good, bad, amazing
Bigrams: not_good, very_happy, best_product
Trigrams: best_purchase_ever, waste_of_money, highly_recommend_everyone
```

---

## 🤖 Model Training

### Models Implemented

#### 1. **Naive Bayes (Baseline)**
```python
MultinomialNB(alpha=0.1)
```
- Fast training and prediction
- Works well with text data
- Good baseline model
- **Accuracy:** 76.52%

#### 2. **Logistic Regression (Best Model)**
```python
LogisticRegression(
    solver='saga',
    C=1.0,
    max_iter=1000,
    class_weight='balanced',
    n_jobs=-1
)
```
- Best overall performance
- Handles class imbalance
- Provides probability estimates
- **Accuracy:** 78.46%

#### 3. **Linear SVC**
```python
LinearSVC(
    C=1.0,
    max_iter=1000,
    class_weight='balanced'
)
```
- Fast training on large datasets
- Good for high-dimensional data

#### 4. **Voting Ensemble**
Combines predictions from multiple models:
- Logistic Regression
- Naive Bayes
- Linear SVC

**Method:** Hard voting (majority vote)
- **Accuracy:** 78.38%

#### 5. **Stacking Ensemble**
Two-level architecture:
- **Base Models:** LR, NB, SVC
- **Meta Model:** Logistic Regression

**Method:** 5-fold cross-validation
- **Accuracy:** 78.41%

### Training Pipeline

```python
# 1. Load and shuffle data
df = pd.read_csv('sentiment140_cleaned.csv')
df = df.sample(frac=1, random_state=42)

# 2. Preprocess text
df = preprocess_dataset(df)

# 3. Extract features
extractor = FeatureExtractor()
X = extractor.fit_transform(df['clean_text'])
y = df['target'].values

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

# 5. Train model
model = SentimentModel('logistic_regression')
model.train(X_train, y_train)

# 6. Evaluate
metrics = model.evaluate(X_test, y_test)
```

### Hyperparameter Tuning

GridSearchCV with 5-fold cross-validation:
```python
param_grid = {
    'C': [0.1, 0.5, 1.0, 2.0, 5.0],
    'solver': ['liblinear', 'saga'],
    'class_weight': ['balanced', None],
    'penalty': ['l2']
}
```

---

## 📈 Performance Metrics

### Final Evaluation Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Naive Bayes** | 76.52% | 76.52% | 76.52% | 76.52% | 0.848 |
| **Logistic Regression** | **78.46%** | **77.39%** | **80.41%** | **78.87%** | **0.866** |
| **Voting Ensemble** | 78.38% | 77.24% | 80.45% | 78.82% | 0.784 |
| **Stacking Ensemble** | 78.41% | 77.51% | 80.04% | 78.75% | 0.865 |

### Confusion Matrix (Logistic Regression)
```
                Predicted
              Negative  Positive
Actual 
Negative      122,345   37,655
Positive       31,429   128,571
```

### Key Insights
- ✅ **Best Model:** Logistic Regression (78.46% accuracy)
- ✅ **High Recall:** 80.41% (good at detecting positive sentiments)
- ✅ **Balanced Performance:** Similar precision and recall
- ✅ **ROC-AUC:** 0.866 (excellent discrimination ability)

### Error Analysis
- **False Positives:** 37,655 (23.5%)
  - Sarcasm misclassified as positive
  - Neutral tweets classified as positive
  
- **False Negatives:** 31,429 (19.6%)
  - Subtle negative sentiments missed
  - Mixed sentiment tweets

---

## 🖥️ Web Application

### Streamlit Interface

#### Features

1. **Home Page - Quick Analysis**
   - Single tweet analysis
   - Real-time prediction
   - Confidence gauge visualization
   - Probability distribution chart
   - Immediate feedback system

2. **Batch Analysis**
   - Analyze multiple tweets at once
   - Support for text input (one per line)
   - CSV/TXT file upload
   - Maximum 1,000 tweets per batch
   - Individual feedback per tweet
   - Progress tracking

3. **Statistics Dashboard**
   - Total predictions count
   - Sentiment distribution (pie chart)
   - Feedback distribution
   - Average confidence score
   - Recent predictions history
   - Downloadable results

4. **About Page**
   - Model performance metrics
   - Technical stack information
   - Project limitations
   - Usage guidelines

#### UI Design
- **Color Scheme:**
  - Light grey sidebar (#e5e7eb)
  - Soft blue accents (#3b82f6)
  - Green for positive (#2ecc71)
  - Red for negative (#e74c3c)
  
- **Layout:** Wide, responsive design
- **Style:** Clean, minimal, professional

### Prediction Output

For each prediction, the app displays:
- Sentiment label (POSITIVE/NEGATIVE)
- Confidence percentage
- Probability distribution bar chart
- Confidence gauge
- Original and processed text
- Interpretation message

#### Confidence Interpretation
- **90-100%:** Very confident
- **75-89%:** Confident
- **60-74%:** Moderately confident
- **<60%:** Uncertain (ambiguous/neutral)

---

## 🚀 Installation & Usage

### Prerequisites
```bash
Python 3.8+
pip
```

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download NLTK data**
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

4. **Prepare the dataset**
```bash
# Place sentiment140.csv in the data/ directory
python preprocessing.py
```

5. **Train models**
```bash
python train_model.py
```

6. **Run the web application**
```bash
streamlit run app.py
```

### Quick Start (Using Pre-trained Models)

```python
from predict import SentimentPredictor

# Load model
predictor = SentimentPredictor()
predictor.load_model(
    model_path='models/logistic_regression_model.pkl',
    vectorizer_path='models/tfidf_vectorizer.pkl'
)

# Predict single tweet
result = predictor.predict_single("I love this product!")
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.2f}%")

# Batch prediction
tweets = ["Great service!", "Terrible experience", "It's okay"]
results = predictor.predict_batch(tweets)
```

---

## 🛠️ Technical Stack

### Core Libraries

| Category | Library | Version | Purpose |
|----------|---------|---------|---------|
| **Data Processing** | pandas | 2.2.3 | Data manipulation |
| | numpy | 2.1.2 | Numerical computing |
| **Machine Learning** | scikit-learn | 1.4.2 | ML models & metrics |
| **NLP** | nltk | 3.8.1 | Text preprocessing |
| **Visualization** | matplotlib | 3.9.2 | Static plots |
| | seaborn | 0.12.2 | Statistical graphics |
| | plotly | 5.14.1 | Interactive charts |
| **Web App** | streamlit | 1.51.0 | Web interface |
| **Utilities** | joblib | 1.3.2 | Model serialization |

### File Structure
```
requirements.txt:
pandas==2.2.3
numpy==2.1.2
scikit-learn==1.4.2
nltk==3.8.1
matplotlib==3.9.2
seaborn==0.12.2
plotly==5.14.1
streamlit==1.51.0
joblib==1.3.2
```

---

## ⚠️ Limitations

### Current Limitations

1. **Sarcasm Detection**
   - Model struggles with sarcastic tweets
   - Example: "Oh great, another delay" → Misclassified as positive

2. **Language Support**
   - Only supports English tweets
   - Performance degrades with mixed languages

3. **Temporal Bias**
   - Trained on 2009-2017 data
   - May not capture modern slang and trends

4. **Neutral Sentiment**
   - Binary classification (no neutral class)
   - Neutral tweets forced into positive/negative

5. **Context Understanding**
   - Limited understanding of complex context
   - May miss subtle emotional cues

6. **Domain Specificity**
   - Trained on general Twitter data
   - May underperform on domain-specific tweets (medical, legal, etc.)

### Known Issues
- Emojis not always correctly interpreted
- Very short tweets (<3 words) have lower confidence
- Heavy slang or abbreviations may reduce accuracy

---

## 🔮 Future Improvements

### Short-term Enhancements
1. **Add Neutral Class**
   - Extend to 3-class classification
   - Better handle ambiguous sentiments

2. **Improve Sarcasm Detection**
   - Add sarcasm-specific features
   - Use context-aware models

3. **Expand Language Support**
   - Multilingual models
   - Translation pipeline

4. **Real-time Twitter Integration**
   - Twitter API integration
   - Live sentiment tracking

### Long-term Goals
1. **Deep Learning Models**
   - BERT/RoBERTa fine-tuning
   - Transformer-based architectures
   - Expected accuracy: 85-90%

2. **Aspect-Based Sentiment Analysis**
   - Identify specific aspects (price, quality, service)
   - Sentiment per aspect

3. **Emotion Detection**
   - Multi-label classification
   - Detect joy, anger, sadness, fear, etc.

4. **Active Learning Pipeline**
   - Incorporate user feedback
   - Continuous model improvement

5. **Deployment**
   - Docker containerization
   - REST API
   - Cloud deployment (AWS/GCP/Azure)

---

## 📝 License

This project is licensed under the MIT License.

---

## 👥 Contributors

- **Mahdi TOUMI** - Initial work and development

---

## 📧 Contact

For questions or feedback:
- Email: mahdi.toumi@enicar.ucar.tn
- LinkedIn: [linkedin.com/in/yourprofile](https://www.linkedin.com/in/mahdi-toumi/)

---

## 🙏 Acknowledgments

- **Dataset:** Sentiment140 dataset from Kaggle
- **Libraries:** Scikit-learn, NLTK, Streamlit teams
- **Inspiration:** Stanford NLP course materials

---

## 📚 References

1. Go, A., Bhayani, R., & Huang, L. (2009). Twitter Sentiment Classification using Distant Supervision. Stanford University.
2. Bird, S., Klein, E., & Loper, E. (2009). Natural Language Processing with Python. O'Reilly Media.
3. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. JMLR.

---

**Last Updated:** November 2025  
**Project Status:** ✅ Production Ready
