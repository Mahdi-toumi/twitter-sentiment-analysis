"""
Model Training Module (Enhanced)
Complete pipeline with Ensemble Methods & Hyperparameter Tuning
Version 2.0 - Optimized for maximum performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier, StackingClassifier
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
import json
from datetime import datetime
from preprocessing import preprocess_dataset, clean_tweet
from feature_extraction import FeatureExtractor


class SentimentModel:
    """
    Wrapper class for sentiment analysis model
    Enhanced with better hyperparameters
    """
    
    def __init__(self, model_type='logistic_regression'):
        """
        Initialize model with optimized hyperparameters
        
        Parameters:
        -----------
        model_type : str
            'naive_bayes', 'logistic_regression', 'svc', 'voting', or 'stacking'
        """
        self.model_type = model_type
        
        if model_type == 'naive_bayes':
            # Optimized Naive Bayes
            self.model = MultinomialNB(
                alpha=0.1  # Reduced from 1.0 for better performance
            )
        elif model_type == 'logistic_regression':
            # Optimized Logistic Regression
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                solver='saga',  # Better for large datasets
                C=1.0,
                class_weight='balanced',  # Handle class imbalance
                n_jobs=-1  # Use all CPU cores
            )
        elif model_type == 'svc':
            # Linear SVC
            self.model = LinearSVC(
                max_iter=1000,
                random_state=42,
                C=1.0,
                class_weight='balanced'
            )
        elif model_type == 'voting':
            # Voting Ensemble
            self.model = self._create_voting_ensemble()
        elif model_type == 'stacking':
            # Stacking Ensemble
            self.model = self._create_stacking_ensemble()
        else:
            raise ValueError("Invalid model_type")
        
        self.is_trained = False
        self.training_time = 0
        self.metrics = {}
    
    def _create_voting_ensemble(self):
        """Create voting ensemble with multiple models"""
        lr = LogisticRegression(max_iter=1000, random_state=42, solver='saga', C=1.0)
        nb = MultinomialNB(alpha=0.1)
        svc = LinearSVC(max_iter=1000, random_state=42, C=1.0)
        
        return VotingClassifier(
            estimators=[('lr', lr), ('nb', nb), ('svc', svc)],
            voting='hard'
        )
    
    def _create_stacking_ensemble(self):
        """Create stacking ensemble"""
        base_models = [
            ('lr', LogisticRegression(max_iter=1000, random_state=42)),
            ('nb', MultinomialNB(alpha=0.1)),
            ('svc', LinearSVC(max_iter=1000, random_state=42))
        ]
        
        meta_model = LogisticRegression(max_iter=1000, random_state=42)
        
        return StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5
        )
        
    def train(self, X_train, y_train):
        """Train the model"""
        print(f"\n{'='*80}")
        print(f"TRAINING {self.model_type.upper()} MODEL")
        print(f"{'='*80}")
        print(f"\nTraining data shape: {X_train.shape}")
        print(f"Training samples: {len(y_train):,}")
        print(f"Positive samples: {sum(y_train == 1):,} ({sum(y_train == 1)/len(y_train)*100:.1f}%)")
        print(f"Negative samples: {sum(y_train == 0):,} ({sum(y_train == 0)/len(y_train)*100:.1f}%)")
        
        print("\nTraining in progress...")
        start_time = time.time()
        
        self.model.fit(X_train, y_train)
        
        self.training_time = time.time() - start_time
        self.is_trained = True
        
        print(f"✅ Training complete in {self.training_time:.2f} seconds")
        
    def evaluate(self, X_test, y_test, show_details=True):
        """Evaluate model performance"""
        if not self.is_trained:
            raise Exception("Model must be trained first!")
        
        print(f"\n{'='*80}")
        print(f"EVALUATING MODEL")
        print(f"{'='*80}")
        
        print("\nMaking predictions on test set...")
        y_pred = self.model.predict(X_test)
        
        # Handle probability predictions
        if hasattr(self.model, 'predict_proba'):
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        elif hasattr(self.model, 'decision_function'):
            y_pred_proba = self.model.decision_function(X_test)
        else:
            y_pred_proba = y_pred.astype(float)
        
        # Calculate metrics
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except:
            roc_auc = 0.0
        
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
            print(f"\n{'Metric':<15} {'Score':<10}")
            print("-"*25)
            for k, v in self.metrics.items():
                print(f"{k.upper():<15} {v:.4f} ({v*100:.2f}%)")
            
            print(f"\n{'='*80}")
            print("DETAILED CLASSIFICATION REPORT")
            print(f"{'='*80}\n")
            print(classification_report(y_test, y_pred, 
                                       target_names=['Negative', 'Positive'],
                                       zero_division=0,
                                       digits=4))
            
            cm = confusion_matrix(y_test, y_pred)
            print(f"{'='*80}")
            print("CONFUSION MATRIX")
            print(f"{'='*80}\n")
            print(f"                  Predicted")
            print(f"                Negative    Positive")
            print(f"Actual  Negative {cm[0,0]:>8,}    {cm[0,1]:>8,}")
            print(f"        Positive {cm[1,0]:>8,}    {cm[1,1]:>8,}")
            
            print(f"\n✅ True Negatives:  {cm[0,0]:,}")
            print(f"✅ True Positives:  {cm[1,1]:,}")
            print(f"❌ False Negatives: {cm[1,0]:,}")
            print(f"❌ False Positives: {cm[0,1]:,}")
        
        return self.metrics
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise Exception("Model must be trained first!")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_trained:
            raise Exception("Model must be trained first!")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[:, 1]
        elif hasattr(self.model, 'decision_function'):
            scores = self.model.decision_function(X)
            # Convert to probabilities using sigmoid
            return 1 / (1 + np.exp(-scores))
        else:
            return self.model.predict(X).astype(float)
    
    def save(self, filepath):
        """Save model to disk"""
        if not self.is_trained:
            raise Exception("Model must be trained first!")
        joblib.dump(self.model, filepath)
        print(f"✅ Model saved to: {filepath}")
    
    def load(self, filepath):
        """Load model from disk"""
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"✅ Model loaded from: {filepath}")


def hyperparameter_tuning(X_train, y_train, X_test, y_test, quick=True):
    """
    Perform hyperparameter tuning with GridSearchCV
    
    Parameters:
    -----------
    quick : bool
        If True, use smaller grid for faster tuning (5 min)
        If False, use comprehensive grid (30-60 min)
    """
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING")
    print("="*80)
    
    if quick:
        print("\n🚀 Quick tuning mode (5-10 minutes)")
        param_grid = {
            'C': [0.5, 1.0, 2.0],
            'solver': ['saga'],
            'class_weight': ['balanced']
        }
    else:
        print("\n⏰ Comprehensive tuning mode (30-60 minutes)")
        param_grid = {
            'C': [0.1, 0.5, 1.0, 2.0, 5.0],
            'solver': ['liblinear', 'saga'],
            'class_weight': ['balanced', None],
            'penalty': ['l2']
        }
    
    print(f"Parameter grid: {param_grid}")
    print(f"Total combinations: {np.prod([len(v) for v in param_grid.values()])}")
    
    base_model = LogisticRegression(max_iter=1000, random_state=42)
    
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=3 if quick else 5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    print("\nSearching for best hyperparameters...")
    start_time = time.time()
    
    grid_search.fit(X_train, y_train)
    
    tuning_time = time.time() - start_time
    
    print(f"\n✅ Tuning complete in {tuning_time:.1f} seconds ({tuning_time/60:.1f} minutes)")
    print(f"\n🏆 Best parameters: {grid_search.best_params_}")
    print(f"🎯 Best CV score: {grid_search.best_score_:.4f} ({grid_search.best_score_*100:.2f}%)")
    
    # Evaluate on test set
    y_pred = grid_search.best_estimator_.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"📊 Test set accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    return grid_search.best_estimator_, grid_search.best_params_


def visualize_performance(y_test, y_pred, y_pred_proba, model_name, save_dir='models/'):
    """Create comprehensive visualization of model performance"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Actual')
    axes[0, 0].set_xlabel('Predicted')
    
    # ROC Curve
    try:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2,
                       label=f'ROC curve (AUC = {roc_auc:.4f})')
        axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                       label='Random Classifier')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve', fontsize=14, fontweight='bold')
        axes[0, 1].legend(loc="lower right")
        axes[0, 1].grid(True, alpha=0.3)
    except:
        axes[0, 1].text(0.5, 0.5, 'ROC curve unavailable', ha='center')
    
    # Prediction Distribution
    try:
        axes[1, 0].hist(y_pred_proba[y_test == 0], bins=50, alpha=0.6,
                       label='Actual Negative', color='red')
        axes[1, 0].hist(y_pred_proba[y_test == 1], bins=50, alpha=0.6,
                       label='Actual Positive', color='green')
        axes[1, 0].set_xlabel('Predicted Probability')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Prediction Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    except:
        axes[1, 0].text(0.5, 0.5, 'Distribution unavailable', ha='center')
    
    # Metrics Bar Chart
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0)
    }
    
    bars = axes[1, 1].bar(metrics.keys(), metrics.values(),
                         color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
    axes[1, 1].set_ylim([0, 1.0])
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Performance Metrics', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom')
    
    plt.suptitle(f'{model_name} - Performance Analysis',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = f"{save_dir}{model_name.lower().replace(' ', '_')}_performance.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualization saved to: {save_path}")


def compare_models(X_train, X_test, y_train, y_test, include_ensemble=True):
    """
    Train and compare multiple models including ensembles
    
    Parameters:
    -----------
    include_ensemble : bool
        Whether to include ensemble methods (takes longer)
    """
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    results = {}
    models_dict = {}
    
    # 1. Naive Bayes
    print("\n" + "="*80)
    print("1. NAIVE BAYES")
    print("="*80)
    nb_model = SentimentModel('naive_bayes')
    nb_model.train(X_train, y_train)
    results['Naive Bayes'] = nb_model.evaluate(X_test, y_test, show_details=True)
    
    y_pred_nb = nb_model.predict(X_test)
    y_pred_proba_nb = nb_model.predict_proba(X_test)
    visualize_performance(y_test, y_pred_nb, y_pred_proba_nb, 'Naive Bayes')
    
    nb_model.save('models/naive_bayes_model.pkl')
    models_dict['naive_bayes'] = nb_model
    
    # 2. Logistic Regression
    print("\n" + "="*80)
    print("2. LOGISTIC REGRESSION")
    print("="*80)
    lr_model = SentimentModel('logistic_regression')
    lr_model.train(X_train, y_train)
    results['Logistic Regression'] = lr_model.evaluate(X_test, y_test, show_details=True)
    
    y_pred_lr = lr_model.predict(X_test)
    y_pred_proba_lr = lr_model.predict_proba(X_test)
    visualize_performance(y_test, y_pred_lr, y_pred_proba_lr, 'Logistic Regression')
    
    lr_model.save('models/logistic_regression_model.pkl')
    models_dict['logistic_regression'] = lr_model
    
    if include_ensemble:
        # 3. Voting Ensemble
        print("\n" + "="*80)
        print("3. VOTING ENSEMBLE")
        print("="*80)
        print("Combining: Logistic Regression + Naive Bayes + Linear SVC")
        
        voting_model = SentimentModel('voting')
        voting_model.train(X_train, y_train)
        results['Voting Ensemble'] = voting_model.evaluate(X_test, y_test, show_details=True)
        
        y_pred_voting = voting_model.predict(X_test)
        y_pred_proba_voting = voting_model.predict_proba(X_test)
        visualize_performance(y_test, y_pred_voting, y_pred_proba_voting, 'Voting Ensemble')
        
        voting_model.save('models/voting_ensemble_model.pkl')
        models_dict['voting_ensemble'] = voting_model
        
        # 4. Stacking Ensemble
        print("\n" + "="*80)
        print("4. STACKING ENSEMBLE")
        print("="*80)
        print("Base models: LR + NB + SVC")
        print("Meta model: Logistic Regression")
        
        stacking_model = SentimentModel('stacking')
        stacking_model.train(X_train, y_train)
        results['Stacking Ensemble'] = stacking_model.evaluate(X_test, y_test, show_details=True)
        
        y_pred_stacking = stacking_model.predict(X_test)
        y_pred_proba_stacking = stacking_model.predict_proba(X_test)
        visualize_performance(y_test, y_pred_stacking, y_pred_proba_stacking, 'Stacking Ensemble')
        
        stacking_model.save('models/stacking_ensemble_model.pkl')
        models_dict['stacking_ensemble'] = stacking_model
    
    # Final Comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    
    comparison_df = pd.DataFrame(results).T
    print("\n" + comparison_df.to_string())
    
    best_model_name = max(results.items(), key=lambda x: x[1]['f1'])[0]
    best_metrics = results[best_model_name]
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   Accuracy:  {best_metrics['accuracy']*100:.2f}%")
    print(f"   F1-Score:  {best_metrics['f1']*100:.2f}%")
    print(f"   ROC-AUC:   {best_metrics['roc_auc']:.4f}")
    
    return models_dict, results


def train_full_pipeline(data_path='data/sentiment140_cleaned.csv', 
                       sample_size=None,
                       test_size=0.2,
                       use_hyperparameter_tuning=False,
                       use_ensemble=True,
                       quick_tuning=True):
    """
    Complete training pipeline with all enhancements
    
    Parameters:
    -----------
    data_path : str
        Path to dataset
    sample_size : int or None
        Number of samples (None = use all)
    test_size : float
        Test set proportion
    use_hyperparameter_tuning : bool
        Whether to perform hyperparameter tuning
    use_ensemble : bool
        Whether to train ensemble models
    quick_tuning : bool
        Quick (5 min) vs comprehensive (60 min) tuning
    """
    print("="*80)
    print("SENTIMENT ANALYSIS - ENHANCED TRAINING PIPELINE")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  • Dataset: {data_path}")
    print(f"  • Sample size: {sample_size if sample_size else 'ALL'}")
    print(f"  • Test size: {test_size*100:.0f}%")
    print(f"  • Hyperparameter tuning: {use_hyperparameter_tuning}")
    print(f"  • Ensemble methods: {use_ensemble}")
    if use_hyperparameter_tuning:
        print(f"  • Tuning mode: {'Quick (5 min)' if quick_tuning else 'Comprehensive (60 min)'}")
    
    start_time = time.time()
    
    # Load data
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)
    
    df = pd.read_csv(data_path, 
                     encoding='latin-1')
    
    print(f"✅ Loaded {len(df):,} samples")

    df['target'] = pd.to_numeric(df['target'], errors='raise')
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print("✅ Data shuffled")
    
    # Sample if needed
    if sample_size:
        df = df.head(sample_size)
        print(f"✅ Using {len(df):,} samples")
    
    # Convert labels
    df['target'] = df['target'].replace(4, 1)
    
    # Preprocess
    print("\n" + "="*80)
    print("STEP 2: PREPROCESSING")
    print("="*80)
    
    df = preprocess_dataset(df, text_column='text')
    df = df[df['clean_text'].str.len() > 0]
    print(f"✅ Preprocessed {len(df):,} tweets")
    
    # Feature extraction
    print("\n" + "="*80)
    print("STEP 3: FEATURE EXTRACTION")
    print("="*80)
    
    extractor = FeatureExtractor()  # Uses enhanced defaults
    X = extractor.fit_transform(df['clean_text'])
    y = df['target'].values
    
    extractor.save('models/tfidf_vectorizer.pkl')
    
    # Split data
    print("\n" + "="*80)
    print("STEP 4: SPLITTING DATA")
    print("="*80)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    
    print(f"✅ Training set: {X_train.shape[0]:,} samples")
    print(f"✅ Test set: {X_test.shape[0]:,} samples")
    
    # Hyperparameter tuning
    if use_hyperparameter_tuning:
        print("\n" + "="*80)
        print("STEP 5: HYPERPARAMETER TUNING")
        print("="*80)
        
        tuned_model, best_params = hyperparameter_tuning(
            X_train, y_train, X_test, y_test, quick=quick_tuning
        )
        
        joblib.dump(tuned_model, 'models/logistic_regression_tuned.pkl')
        
        with open('models/best_hyperparameters.json', 'w') as f:
            json.dump(best_params, f, indent=2)
        
        print("✅ Tuned model saved")
    
    # Train models
    print("\n" + "="*80)
    print("STEP 6: TRAINING MODELS")
    print("="*80)
    
    models_dict, results = compare_models(X_train, X_test, y_train, y_test, 
                                         include_ensemble=use_ensemble)
    
    # Test examples
    print("\n" + "="*80)
    print("STEP 7: TESTING WITH EXAMPLES")
    print("="*80)
    
    test_examples = [
        "I absolutely love this product! Best purchase ever!",
        "Terrible quality. Very disappointed. Would not recommend.",
        "It's okay, nothing special.",
        "Amazing service! Highly recommend to everyone!",
        "Waste of money. Complete garbage."
    ]
    
    # Use best model
    best_model_name = max(results.items(), key=lambda x: x[1]['f1'])[0]
    best_model_key = best_model_name.lower().replace(' ', '_')
    best_model = models_dict[best_model_key]
    
    print(f"\nUsing best model: {best_model_name}")
    print("-"*80)
    
    for example in test_examples:
        clean = clean_tweet(example)
        X_example = extractor.transform([clean])
        pred = best_model.predict(X_example)[0]
        proba = best_model.predict_proba(X_example)[0]
        
        sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
        confidence = proba * 100
        
        print(f"\nTweet: {example}")
        print(f"Cleaned: {clean}")
        print(f"→ {sentiment} ({confidence:.1f}% confidence)")
    
    # Save metadata
    total_time = time.time() - start_time
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'training_samples': len(df),
        'test_samples': len(y_test),
        'vocabulary_size': extractor.vocabulary_size,
        'training_time_seconds': total_time,
        'best_model': best_model_name,
        'results': {k: {m: float(v) for m, v in metrics.items()} 
                   for k, metrics in results.items()}
    }
    
    with open('models/training_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Final summary
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\n⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"\n🏆 Best model: {best_model_name}")
    print(f"   Accuracy: {results[best_model_name]['accuracy']*100:.2f}%")
    print(f"   F1-Score: {results[best_model_name]['f1']*100:.2f}%")
    
    print("\n📁 Saved files:")
    print("   • models/naive_bayes_model.pkl")
    print("   • models/logistic_regression_model.pkl")
    if use_ensemble:
        print("   • models/voting_ensemble_model.pkl")
        print("   • models/stacking_ensemble_model.pkl")
    if use_hyperparameter_tuning:
        print("   • models/logistic_regression_tuned.pkl")
        print("   • models/best_hyperparameters.json")
    print("   • models/tfidf_vectorizer.pkl")
    print("   • models/training_metadata.json")
    
    return models_dict, extractor, results


if __name__ == "__main__":
    # Example usage with all enhancements
    train_full_pipeline(
        data_path='data/sentiment140_cleaned.csv',
        sample_size=100000,  # Use 100K for testing, None for full dataset
        test_size=0.2,
        use_hyperparameter_tuning=False,  # Set True for tuning (adds 5-60 min)
        use_ensemble=True,  # Train ensemble models
        quick_tuning=True  # Quick tuning if enabled
    )