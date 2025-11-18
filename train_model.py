"""
Model Training Module
Complete pipeline for training sentiment analysis models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)
import joblib
import time
from preprocessing import preprocess_dataset, clean_tweet
from feature_extraction import FeatureExtractor


class SentimentModel:
    """
    Wrapper class for sentiment analysis model
    """
    
    def __init__(self, model_type='naive_bayes'):
        self.model_type = model_type
        
        if model_type == 'naive_bayes':
            self.model = MultinomialNB(alpha=1.0)
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(max_iter=1000, random_state=42, solver='liblinear')
        else:
            raise ValueError("model_type must be 'naive_bayes' or 'logistic_regression'")
        
        self.is_trained = False
        self.training_time = 0
        self.metrics = {}
        
    def train(self, X_train, y_train):
        print(f"\n{'='*80}")
        print(f"TRAINING {self.model_type.upper()} MODEL")
        print(f"{'='*80}")
        print(f"\nTraining data shape: {X_train.shape}")
        print(f"Training samples: {len(y_train)}")
        print(f"Positive samples: {sum(y_train == 1)} ({sum(y_train == 1)/len(y_train)*100:.1f}%)")
        print(f"Negative samples: {sum(y_train == 0)} ({sum(y_train == 0)/len(y_train)*100:.1f}%)")
        
        print("\nTraining in progress...")
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time
        self.is_trained = True
        print(f"✅ Training complete in {self.training_time:.2f} seconds")
        
    def evaluate(self, X_test, y_test, show_details=True):
        if not self.is_trained:
            raise Exception("Model must be trained first!")
        
        print(f"\n{'='*80}")
        print(f"EVALUATING MODEL")
        print(f"{'='*80}")
        print("\nMaking predictions on test set...")
        y_pred = self.model.predict(X_test)
        
        # Handle case where only one class is present
        if len(self.model.classes_) == 1:
            single_class = self.model.classes_[0]
            y_pred_proba = np.ones(len(y_test)) if single_class == 1 else np.zeros(len(y_test))
        else:
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except ValueError:
            roc_auc = 0.0  # fallback if only one class in y_test
        
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc
        }
        
        if show_details:
            print(f"\n{'='*80}")
            print("PERFORMANCE METRICS")
            print(f"{'='*80}")
            for k, v in self.metrics.items():
                print(f"{k:<12s}: {v:.4f}")
            print("\n" + classification_report(y_test, y_pred, target_names=['Negative', 'Positive'], zero_division=0))
            cm = confusion_matrix(y_test, y_pred)
            print("\nConfusion Matrix:\n", cm)
        
        return self.metrics
    
    def predict(self, X):
        if not self.is_trained:
            raise Exception("Model must be trained first!")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if not self.is_trained:
            raise Exception("Model must be trained first!")
        return self.model.predict_proba(X)[:, 1]
    
    def save(self, filepath):
        if not self.is_trained:
            raise Exception("Model must be trained first!")
        joblib.dump(self.model, filepath)
        print(f"✅ Model saved to: {filepath}")
    
    def load(self, filepath):
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"✅ Model loaded from: {filepath}")


def visualize_performance(y_test, y_pred, y_pred_proba, model_name, save_dir='models/'):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
    axes[0,0].set_title('Confusion Matrix')
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    axes[0,1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    axes[0,1].plot([0,1],[0,1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    axes[1,0].hist(y_pred_proba[y_test==0], bins=50, alpha=0.6, label='Actual Negative', color='red')
    axes[1,0].hist(y_pred_proba[y_test==1], bins=50, alpha=0.6, label='Actual Positive', color='green')
    axes[1,1].bar(['Accuracy','Precision','Recall','F1-Score','ROC-AUC'], [accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, zero_division=0), recall_score(y_test, y_pred, zero_division=0), f1_score(y_test, y_pred, zero_division=0), roc_auc], color=['#3498db','#e74c3c','#2ecc71','#f39c12','#9b59b6'])
    save_path = f"{save_dir}{model_name.lower().replace(' ','_')}_performance.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Performance visualization saved to: {save_path}")


def compare_models(X_train, X_test, y_train, y_test):
    results = {}
    nb_model = SentimentModel('naive_bayes')
    nb_model.train(X_train, y_train)
    results['Naive Bayes'] = nb_model.evaluate(X_test, y_test, show_details=True)
    visualize_performance(y_test, nb_model.predict(X_test), nb_model.predict_proba(X_test), 'Naive Bayes')
    nb_model.save('models/naive_bayes_model.pkl')
    
    lr_model = SentimentModel('logistic_regression')
    lr_model.train(X_train, y_train)
    results['Logistic Regression'] = lr_model.evaluate(X_test, y_test, show_details=True)
    visualize_performance(y_test, lr_model.predict(X_test), lr_model.predict_proba(X_test), 'Logistic Regression')
    lr_model.save('models/logistic_regression_model.pkl')
    
    return nb_model, lr_model, results


def train_full_pipeline(data_path='data/sentiment140_cleaned.csv', sample_size=None, test_size=0.2):
    print("="*80)
    print("SENTIMENT ANALYSIS - COMPLETE TRAINING PIPELINE")
    print("="*80)
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df):,} samples")
    
    # Shuffle data to avoid single-class sampling
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print("✅ Data shuffled")
    
    # Sample if needed
    if sample_size:
        df = df.head(sample_size)
        print(f"✅ Sampled {len(df):,} shuffled samples")
    
    df = preprocess_dataset(df, text_column='text')
    df = df[df['clean_text'].str.len() > 0]
    
    extractor = FeatureExtractor(max_features=5000)
    X = extractor.fit_transform(df['clean_text'])
    y = df['target'].values
    extractor.save('models/tfidf_vectorizer.pkl')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    
    nb_model, lr_model, results = compare_models(X_train, X_test, y_train, y_test)
    
    # Test examples
    test_examples = [
        "I absolutely love this product! Best purchase ever!",
        "Terrible quality. Very disappointed. Would not recommend.",
        "It's okay, nothing special.",
        "Amazing service! Highly recommend to everyone!",
        "Waste of money. Complete garbage."
    ]
    
    for example in test_examples:
        clean = clean_tweet(example)
        X_example = extractor.transform([clean])
        pred = nb_model.predict(X_example)[0]
        proba = nb_model.predict_proba(X_example)[0]
        sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
        confidence = proba * 100
        print(f"\nTweet: {example}\nPrediction: {sentiment} (confidence: {confidence:.1f}%)")
    
    return nb_model, lr_model, extractor, results


if __name__ == "__main__":
    train_full_pipeline(data_path='data/sentiment140_cleaned.csv', sample_size=100000, test_size=0.2)
