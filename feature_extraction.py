"""
Feature Extraction Module
Converts cleaned text into numerical features using TF-IDF
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import matplotlib.pyplot as plt


class FeatureExtractor:
    """
    Handles conversion of text to numerical features
    """
    
    def __init__(self, max_features=5000, max_df=0.9, min_df=5):
        """
        Initialize TF-IDF vectorizer
        
        Parameters:
        -----------
        max_features : int
            Maximum number of words to keep (vocabulary size)
            Higher = more detailed but slower
            
        max_df : float
            Ignore words that appear in more than max_df% of documents
            Removes very common words like "the", "is"
            
        min_df : int
            Ignore words that appear in fewer than min_df documents
            Removes very rare words/typos
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            max_df=max_df,
            min_df=min_df,
            ngram_range=(1, 2),  # Use unigrams and bigrams
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True
        )
        
        self.feature_names = None
        self.vocabulary_size = 0
        
    def fit_transform(self, texts):
        """
        Learn vocabulary and transform texts to TF-IDF features
        
        Parameters:
        -----------
        texts : list or Series
            Cleaned text data
            
        Returns:
        --------
        X : sparse matrix
            TF-IDF features (each row is a tweet, each column is a word)
        """
        print("Learning vocabulary from texts...")
        X = self.vectorizer.fit_transform(texts)
        
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.vocabulary_size = len(self.feature_names)
        
        print(f"✅ Vocabulary learned: {self.vocabulary_size} features")
        print(f"✅ Feature matrix shape: {X.shape}")
        print(f"   ({X.shape[0]} tweets × {X.shape[1]} features)")
        
        return X
    
    def transform(self, texts):
        """
        Transform new texts using learned vocabulary
        
        Parameters:
        -----------
        texts : list or Series
            New cleaned text data
            
        Returns:
        --------
        X : sparse matrix
            TF-IDF features
        """
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self):
        """Get list of feature names (words in vocabulary)"""
        return self.feature_names
    
    def get_top_features_for_text(self, text, top_n=10):
        """
        Get top N most important features for a given text
        
        Parameters:
        -----------
        text : str
            Cleaned text
        top_n : int
            Number of top features to return
            
        Returns:
        --------
        list of tuples : (feature_name, tfidf_score)
        """
        # Transform text
        vector = self.vectorizer.transform([text])
        
        # Get non-zero features
        indices = vector.nonzero()[1]
        scores = vector.data
        
        # Get feature names and scores
        features = [(self.feature_names[i], scores[idx]) 
                   for idx, i in enumerate(indices)]
        
        # Sort by score
        features = sorted(features, key=lambda x: x[1], reverse=True)
        
        return features[:top_n]
    
    def visualize_vocabulary_distribution(self, save_path='vocabulary_distribution.png'):
        """
        Visualize how features are distributed
        """
        if self.feature_names is None:
            print("❌ Error: Vectorizer not fitted yet!")
            return
        
        # Get IDF values
        idf_scores = self.vectorizer.idf_
        
        plt.figure(figsize=(12, 6))
        
        # Subplot 1: IDF distribution
        plt.subplot(1, 2, 1)
        plt.hist(idf_scores, bins=50, color='skyblue', edgecolor='black')
        plt.xlabel('IDF Score', fontsize=12)
        plt.ylabel('Number of Words', fontsize=12)
        plt.title('Distribution of IDF Scores', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Top 20 words by IDF
        plt.subplot(1, 2, 2)
        top_20_indices = np.argsort(idf_scores)[-20:]
        top_20_words = [self.feature_names[i] for i in top_20_indices]
        top_20_scores = idf_scores[top_20_indices]
        
        plt.barh(range(20), top_20_scores, color='coral')
        plt.yticks(range(20), top_20_words)
        plt.xlabel('IDF Score', fontsize=12)
        plt.title('Top 20 Words by IDF Score', fontsize=14)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved to: {save_path}")
        plt.close()
    
    def save(self, filepath):
        """Save vectorizer to disk"""
        joblib.dump(self.vectorizer, filepath)
        print(f"✅ Vectorizer saved to: {filepath}")
    
    def load(self, filepath):
        """Load vectorizer from disk"""
        self.vectorizer = joblib.load(filepath)
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.vocabulary_size = len(self.feature_names)
        print(f"✅ Vectorizer loaded from: {filepath}")
        print(f"   Vocabulary size: {self.vocabulary_size}")


def demonstrate_feature_extraction():
    """
    Demonstrate how feature extraction works with examples
    """
    print("=" * 80)
    print("FEATURE EXTRACTION DEMONSTRATION")
    print("=" * 80)
    
    # Sample tweets (already cleaned)
    sample_tweets = [
        "love product amaz",
        "hate product terribl",
        "love love love best",
        "disappoint bad qualiti",
        "great recommend everyon"
    ]
    
    # Create feature extractor for demo
    extractor = FeatureExtractor(max_features=20, min_df=1, max_df=1.0)
    
    # Fit and transform
    X = extractor.fit_transform(sample_tweets)
    
    print(f"\nVocabulary: {list(extractor.get_feature_names())}")
    
    # Show features for each tweet
    print("\n" + "-" * 80)
    print("FEATURES FOR EACH TWEET:")
    print("-" * 80)
    
    for i, tweet in enumerate(sample_tweets):
        print(f"\nTweet {i+1}: \"{tweet}\"")
        features = extractor.get_top_features_for_text(tweet, top_n=5)
        print("Top features:")
        for feature, score in features:
            print(f"  - {feature:15s}: {score:.4f}")
    
    return extractor, X


if __name__ == "__main__":
    demonstrate_feature_extraction()