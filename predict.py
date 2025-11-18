"""
Prediction Module
Complete system for making sentiment predictions on new tweets
"""

import joblib
import numpy as np
import pandas as pd
from preprocessing import clean_tweet
import os
import warnings
warnings.filterwarnings('ignore')


class SentimentPredictor:
    """
    Complete prediction system for sentiment analysis
    Handles loading models, preprocessing, and making predictions
    """
    
    def __init__(self, model_path=None, vectorizer_path=None):
        """
        Initialize the predictor
        
        Parameters:
        -----------
        model_path : str
            Path to saved model (.pkl file)
        vectorizer_path : str
            Path to saved vectorizer (.pkl file)
        """
        self.model = None
        self.vectorizer = None
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.is_loaded = False
        
        # Auto-load if paths provided
        if model_path and vectorizer_path:
            self.load_model(model_path, vectorizer_path)
    
    def load_model(self, model_path, vectorizer_path):
        """
        Load trained model and vectorizer from disk
        
        Parameters:
        -----------
        model_path : str
            Path to model file
        vectorizer_path : str
            Path to vectorizer file
        """
        print("="*80)
        print("LOADING MODEL")
        print("="*80)
        
        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")
        
        # Load model
        print(f"\nLoading model from: {model_path}")
        self.model = joblib.load(model_path)
        print("✅ Model loaded successfully")
        
        # Load vectorizer
        print(f"\nLoading vectorizer from: {vectorizer_path}")
        self.vectorizer = joblib.load(vectorizer_path)
        print("✅ Vectorizer loaded successfully")
        
        # Get model info
        model_type = type(self.model).__name__
        vocab_size = len(self.vectorizer.get_feature_names_out())
        
        print(f"\n{'='*80}")
        print("MODEL INFORMATION")
        print(f"{'='*80}")
        print(f"Model Type: {model_type}")
        print(f"Vocabulary Size: {vocab_size:,} features")
        
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.is_loaded = True
        
        print(f"\n✅ Predictor ready for use!")
    
    def predict_single(self, tweet, return_proba=True):
        """
        Predict sentiment for a single tweet
        
        Parameters:
        -----------
        tweet : str
            Raw tweet text
        return_proba : bool
            Whether to return probability scores
            
        Returns:
        --------
        dict : Prediction results
            {
                'original_tweet': str,
                'cleaned_tweet': str,
                'sentiment': str ('POSITIVE' or 'NEGATIVE'),
                'confidence': float (0-100),
                'probabilities': dict {'positive': float, 'negative': float}
            }
        """
        if not self.is_loaded:
            raise Exception("Model not loaded! Call load_model() first.")
        
        # Step 1: Clean the tweet
        cleaned = clean_tweet(tweet)
        
        # Handle empty tweet after cleaning
        if not cleaned or len(cleaned.strip()) == 0:
            return {
                'original_tweet': tweet,
                'cleaned_tweet': cleaned,
                'sentiment': 'NEUTRAL',
                'confidence': 0.0,
                'probabilities': {'positive': 0.5, 'negative': 0.5},
                'error': 'Tweet is empty after preprocessing'
            }
        
        # Step 2: Vectorize
        X = self.vectorizer.transform([cleaned])
        
        # Step 3: Predict
        prediction = self.model.predict(X)[0]
        
        # Step 4: Get probabilities
        if return_proba:
            proba = self.model.predict_proba(X)[0]
            prob_negative = proba[0]
            prob_positive = proba[1]
            confidence = max(prob_negative, prob_positive) * 100
        else:
            prob_negative = None
            prob_positive = None
            confidence = None
        
        # Step 5: Format results
        sentiment = 'POSITIVE' if prediction == 1 else 'NEGATIVE'
        
        result = {
            'original_tweet': tweet,
            'cleaned_tweet': cleaned,
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': {
                'positive': prob_positive,
                'negative': prob_negative
            }
        }
        
        return result
    
    def predict_batch(self, tweets, show_progress=True):
        """
        Predict sentiment for multiple tweets
        
        Parameters:
        -----------
        tweets : list of str
            List of raw tweets
        show_progress : bool
            Whether to show progress
            
        Returns:
        --------
        list of dict : Prediction results for each tweet
        """
        if not self.is_loaded:
            raise Exception("Model not loaded! Call load_model() first.")
        
        results = []
        total = len(tweets)
        
        if show_progress:
            print(f"\nPredicting sentiment for {total} tweets...")
        
        for i, tweet in enumerate(tweets):
            result = self.predict_single(tweet, return_proba=True)
            results.append(result)
            
            if show_progress and (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{total} tweets...")
        
        if show_progress:
            print(f"✅ Completed {total} predictions")
        
        return results
    
    def predict_dataframe(self, df, text_column='text'):
        """
        Predict sentiment for tweets in a DataFrame
        
        Parameters:
        -----------
        df : pandas DataFrame
            DataFrame containing tweets
        text_column : str
            Name of column containing tweet text
            
        Returns:
        --------
        pandas DataFrame : Original DataFrame with added columns:
            - sentiment
            - confidence
            - prob_positive
            - prob_negative
        """
        if not self.is_loaded:
            raise Exception("Model not loaded! Call load_model() first.")
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        print(f"\nPredicting sentiment for {len(df)} tweets in DataFrame...")
        
        # Make predictions
        results = self.predict_batch(df[text_column].tolist(), show_progress=True)
        
        # Add results to DataFrame
        df['sentiment'] = [r['sentiment'] for r in results]
        df['confidence'] = [r['confidence'] for r in results]
        df['prob_positive'] = [r['probabilities']['positive'] for r in results]
        df['prob_negative'] = [r['probabilities']['negative'] for r in results]
        
        print("✅ Predictions added to DataFrame")
        
        return df
    
    def explain_prediction(self, tweet, top_n=10):
        """
        Explain why the model made a particular prediction
        Shows most important words that influenced the decision
        
        Parameters:
        -----------
        tweet : str
            Raw tweet text
        top_n : int
            Number of top features to show
            
        Returns:
        --------
        dict : Explanation with top influential words
        """
        if not self.is_loaded:
            raise Exception("Model not loaded! Call load_model() first.")
        
        # Get prediction
        result = self.predict_single(tweet)
        
        # Clean tweet
        cleaned = clean_tweet(tweet)
        
        # Vectorize
        X = self.vectorizer.transform([cleaned])
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get non-zero features (words present in tweet)
        nonzero_indices = X.nonzero()[1]
        nonzero_features = [(feature_names[i], X[0, i]) for i in nonzero_indices]
        
        # Sort by TF-IDF score
        nonzero_features = sorted(nonzero_features, key=lambda x: x[1], reverse=True)
        
        # Get model coefficients (for Logistic Regression)
        if hasattr(self.model, 'coef_'):
            # Logistic Regression
            coefficients = self.model.coef_[0]
            
            # Get feature importance
            feature_importance = []
            for feature, tfidf_score in nonzero_features:
                feature_idx = list(feature_names).index(feature)
                coefficient = coefficients[feature_idx]
                importance = tfidf_score * coefficient
                
                feature_importance.append({
                    'word': feature,
                    'tfidf_score': tfidf_score,
                    'coefficient': coefficient,
                    'importance': importance,
                    'direction': 'positive' if coefficient > 0 else 'negative'
                })
            
            # Sort by absolute importance
            feature_importance = sorted(feature_importance, 
                                      key=lambda x: abs(x['importance']), 
                                      reverse=True)[:top_n]
        else:
            # Naive Bayes - use TF-IDF scores
            feature_importance = [
                {
                    'word': word,
                    'tfidf_score': score,
                    'importance': score
                }
                for word, score in nonzero_features[:top_n]
            ]
        
        explanation = {
            'prediction': result,
            'top_features': feature_importance,
            'num_features_used': len(nonzero_indices)
        }
        
        return explanation
    
    def get_statistics(self, predictions):
        """
        Calculate statistics from a batch of predictions
        
        Parameters:
        -----------
        predictions : list of dict
            Results from predict_batch()
            
        Returns:
        --------
        dict : Statistics summary
        """
        total = len(predictions)
        positive = sum(1 for p in predictions if p['sentiment'] == 'POSITIVE')
        negative = total - positive
        
        confidences = [p['confidence'] for p in predictions]
        avg_confidence = np.mean(confidences)
        
        stats = {
            'total_tweets': total,
            'positive_count': positive,
            'negative_count': negative,
            'positive_percentage': (positive / total) * 100,
            'negative_percentage': (negative / total) * 100,
            'average_confidence': avg_confidence,
            'high_confidence': sum(1 for c in confidences if c > 90),
            'medium_confidence': sum(1 for c in confidences if 70 <= c <= 90),
            'low_confidence': sum(1 for c in confidences if c < 70)
        }
        
        return stats


def demonstrate_predictions():
    """
    Demonstrate the prediction system with examples
    """
    print("="*80)
    print("PREDICTION SYSTEM DEMONSTRATION")
    print("="*80)
    
    # Initialize predictor
    predictor = SentimentPredictor()
    
    # Load models
    try:
        predictor.load_model(
            model_path='models/logistic_regression_model.pkl',
            vectorizer_path='models/tfidf_vectorizer.pkl'
        )
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease train the model first by running: python run_training.py")
        return
    
    # Test tweets
    test_tweets = [
        "I absolutely love this product! Best purchase ever! 😍",
        "Terrible quality. Very disappointed. Would not recommend. 😠",
        "It's okay, nothing special really.",
        "Amazing customer service! They went above and beyond!",
        "Waste of money. Complete garbage. Save your money.",
        "Pretty good, would buy again.",
        "Not bad but could be better.",
        "Worst experience ever! Never shopping here again!",
        "Highly recommend! You won't regret it!",
        "Meh, it's average at best."
    ]
    
    print("\n" + "="*80)
    print("SINGLE TWEET PREDICTIONS")
    print("="*80)
    
    for i, tweet in enumerate(test_tweets, 1):
        print(f"\n{'-'*80}")
        print(f"Tweet {i}: {tweet}")
        print(f"{'-'*80}")
        
        result = predictor.predict_single(tweet)
        
        print(f"Cleaned: {result['cleaned_tweet']}")
        print(f"\n🎯 Sentiment: {result['sentiment']}")
        print(f"📊 Confidence: {result['confidence']:.1f}%")
        print(f"📈 Probabilities:")
        print(f"   Positive: {result['probabilities']['positive']*100:.1f}%")
        print(f"   Negative: {result['probabilities']['negative']*100:.1f}%")
    
    # Batch prediction
    print("\n" + "="*80)
    print("BATCH PREDICTION STATISTICS")
    print("="*80)
    
    results = predictor.predict_batch(test_tweets, show_progress=False)
    stats = predictor.get_statistics(results)
    
    print(f"\nTotal Tweets Analyzed: {stats['total_tweets']}")
    print(f"\n📊 Sentiment Distribution:")
    print(f"   Positive: {stats['positive_count']} ({stats['positive_percentage']:.1f}%)")
    print(f"   Negative: {stats['negative_count']} ({stats['negative_percentage']:.1f}%)")
    
    print(f"\n📈 Confidence Distribution:")
    print(f"   High (>90%):   {stats['high_confidence']} tweets")
    print(f"   Medium (70-90%): {stats['medium_confidence']} tweets")
    print(f"   Low (<70%):    {stats['low_confidence']} tweets")
    print(f"\n   Average Confidence: {stats['average_confidence']:.1f}%")
    
    # Explain a prediction
    print("\n" + "="*80)
    print("PREDICTION EXPLANATION")
    print("="*80)
    
    explain_tweet = test_tweets[0]  # "I absolutely love this product..."
    print(f"\nExplaining prediction for:")
    print(f'"{explain_tweet}"')
    
    explanation = predictor.explain_prediction(explain_tweet, top_n=5)
    
    print(f"\n🎯 Prediction: {explanation['prediction']['sentiment']}")
    print(f"📊 Confidence: {explanation['prediction']['confidence']:.1f}%")
    print(f"\n🔍 Top 5 Most Influential Words:")
    
    for i, feature in enumerate(explanation['top_features'], 1):
        if 'direction' in feature:
            direction = '🟢' if feature['direction'] == 'positive' else '🔴'
            print(f"{i}. {direction} '{feature['word']}'")
            print(f"   TF-IDF Score: {feature['tfidf_score']:.4f}")
            print(f"   Coefficient: {feature['coefficient']:.4f}")
            print(f"   Impact: {feature['importance']:.4f}")
        else:
            print(f"{i}. '{feature['word']}'")
            print(f"   TF-IDF Score: {feature['tfidf_score']:.4f}")


if __name__ == "__main__":
    demonstrate_predictions()