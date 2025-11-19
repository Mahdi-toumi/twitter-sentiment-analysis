"""
Feature Extraction Module (Enhanced)
Converts cleaned text into numerical features using TF-IDF
Version 2.0 - Improved for better performance
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import matplotlib.pyplot as plt


class FeatureExtractor:
    """
    Handles conversion of text to numerical features
    Enhanced version with better default parameters
    """
    
    def __init__(self, max_features=10000, max_df=0.8, min_df=3, ngram_range=(1, 3)):
        """
        Initialize TF-IDF vectorizer with enhanced settings
        
        Parameters:
        -----------
        max_features : int, default=10000
            Maximum number of words to keep (vocabulary size)
            Increased from 5000 for better feature coverage
            Higher = more detailed patterns captured
            
        max_df : float, default=0.8
            Ignore words that appear in more than max_df% of documents
            Reduced from 0.9 to filter more common words
            Removes very common words like "the", "is", "and"
            
        min_df : int, default=3
            Ignore words that appear in fewer than min_df documents
            Reduced from 5 to keep more rare but meaningful words
            Removes very rare words/typos while keeping useful terms
            
        ngram_range : tuple, default=(1, 3)
            Range of n-grams to extract
            (1, 3) = unigrams + bigrams + trigrams
            Captures phrases like "not good", "very happy", "best ever seen"
        
        Changes from v1.0:
        ------------------
        ✅ max_features: 5000 → 10000 (2x vocabulary)
        ✅ max_df: 0.9 → 0.8 (stricter filtering)
        ✅ min_df: 5 → 3 (keep more rare words)
        ✅ ngram_range: (1,2) → (1,3) (capture longer phrases)
        ✅ Added sublinear_tf=True (reduces impact of very frequent words)
        ✅ Better token pattern
        """
        self.vectorizer = TfidfVectorizer(
            # Vocabulary size
            max_features=max_features,
            
            # Filtering parameters
            max_df=max_df,
            min_df=min_df,
            
            # N-grams: unigrams, bigrams, trigrams
            ngram_range=ngram_range,
            
            # Text processing
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\b\w+\b',  # Better word boundary detection
            lowercase=True,  # Ensure lowercase (redundant but safe)
            
            # TF-IDF parameters
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True,  # ← NEW! Use log(TF) instead of TF
            
            # Normalization
            norm='l2',  # L2 normalization
            
            # Performance
            dtype=np.float32  # Use float32 for memory efficiency
        )
        
        self.feature_names = None
        self.vocabulary_size = 0
        self.ngram_range = ngram_range
        self.max_features = max_features
        
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
        print("="*80)
        print("FEATURE EXTRACTION - ENHANCED")
        print("="*80)
        print(f"\nConfiguration:")
        print(f"  • N-gram range: {self.vectorizer.ngram_range}")
        print(f"  • Max features: {self.vectorizer.max_features:,}")
        print(f"  • Max DF: {self.vectorizer.max_df}")
        print(f"  • Min DF: {self.vectorizer.min_df}")
        print(f"  • Sublinear TF: {self.vectorizer.sublinear_tf}")
        print(f"\nLearning vocabulary from {len(texts):,} texts...")
        
        X = self.vectorizer.fit_transform(texts)
        
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.vocabulary_size = len(self.feature_names)
        
        # Calculate sparsity
        sparsity = 1.0 - (X.nnz / (X.shape[0] * X.shape[1]))
        
        print(f"\n✅ Vocabulary learned: {self.vocabulary_size:,} features")
        print(f"✅ Feature matrix shape: {X.shape}")
        print(f"   ({X.shape[0]:,} tweets × {X.shape[1]:,} features)")
        print(f"✅ Matrix sparsity: {sparsity*100:.2f}%")
        print(f"   (Only {(1-sparsity)*100:.2f}% of values are non-zero)")
        
        # Show sample n-grams
        self._show_sample_ngrams()
        
        return X
    
    def _show_sample_ngrams(self):
        """Show sample n-grams extracted"""
        if self.feature_names is None:
            return
        
        print(f"\n📝 Sample N-grams Extracted:")
        
        # Unigrams
        unigrams = [f for f in self.feature_names if len(f.split()) == 1]
        if unigrams:
            print(f"  Unigrams (1 word): {', '.join(unigrams[:5])} ...")
        
        # Bigrams
        bigrams = [f for f in self.feature_names if len(f.split()) == 2]
        if bigrams:
            print(f"  Bigrams (2 words): {', '.join(bigrams[:5])} ...")
        
        # Trigrams
        trigrams = [f for f in self.feature_names if len(f.split()) == 3]
        if trigrams:
            print(f"  Trigrams (3 words): {', '.join(trigrams[:5])} ...")
        
        print(f"\n  Total: {len(unigrams):,} unigrams + {len(bigrams):,} bigrams + {len(trigrams):,} trigrams")
    
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
    
    def get_vocabulary_stats(self):
        """
        Get statistics about the vocabulary
        
        Returns:
        --------
        dict : Statistics about features
        """
        if self.feature_names is None:
            return {"error": "Vectorizer not fitted yet"}
        
        # Count n-grams
        unigrams = sum(1 for f in self.feature_names if len(f.split()) == 1)
        bigrams = sum(1 for f in self.feature_names if len(f.split()) == 2)
        trigrams = sum(1 for f in self.feature_names if len(f.split()) == 3)
        
        stats = {
            'total_features': self.vocabulary_size,
            'unigrams': unigrams,
            'bigrams': bigrams,
            'trigrams': trigrams,
            'ngram_range': self.ngram_range,
            'max_features': self.max_features,
            'idf_mean': float(np.mean(self.vectorizer.idf_)),
            'idf_std': float(np.std(self.vectorizer.idf_)),
            'idf_min': float(np.min(self.vectorizer.idf_)),
            'idf_max': float(np.max(self.vectorizer.idf_))
        }
        
        return stats
    
    def visualize_vocabulary_distribution(self, save_path='vocabulary_distribution.png'):
        """
        Visualize how features are distributed
        Enhanced version with more insights
        """
        if self.feature_names is None:
            print("❌ Error: Vectorizer not fitted yet!")
            return
        
        # Get IDF values
        idf_scores = self.vectorizer.idf_
        
        # Categorize by n-gram type
        unigram_idfs = []
        bigram_idfs = []
        trigram_idfs = []
        
        for i, feature in enumerate(self.feature_names):
            n = len(feature.split())
            if n == 1:
                unigram_idfs.append(idf_scores[i])
            elif n == 2:
                bigram_idfs.append(idf_scores[i])
            elif n == 3:
                trigram_idfs.append(idf_scores[i])
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Subplot 1: Overall IDF distribution
        axes[0, 0].hist(idf_scores, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(np.mean(idf_scores), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(idf_scores):.2f}')
        axes[0, 0].set_xlabel('IDF Score', fontsize=12)
        axes[0, 0].set_ylabel('Number of Features', fontsize=12)
        axes[0, 0].set_title('Distribution of IDF Scores', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Subplot 2: IDF by n-gram type
        axes[0, 1].boxplot([unigram_idfs, bigram_idfs, trigram_idfs],
                           labels=['Unigrams', 'Bigrams', 'Trigrams'])
        axes[0, 1].set_ylabel('IDF Score', fontsize=12)
        axes[0, 1].set_title('IDF Distribution by N-gram Type', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Subplot 3: Top 20 features by IDF
        top_20_indices = np.argsort(idf_scores)[-20:]
        top_20_words = [self.feature_names[i] for i in top_20_indices]
        top_20_scores = idf_scores[top_20_indices]
        
        colors = ['coral' if len(w.split()) == 1 else 'lightblue' if len(w.split()) == 2 
                 else 'lightgreen' for w in top_20_words]
        
        axes[1, 0].barh(range(20), top_20_scores, color=colors)
        axes[1, 0].set_yticks(range(20))
        axes[1, 0].set_yticklabels(top_20_words, fontsize=9)
        axes[1, 0].set_xlabel('IDF Score', fontsize=12)
        axes[1, 0].set_title('Top 20 Features by IDF Score', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # Subplot 4: Feature count by n-gram type
        ngram_counts = [len(unigram_idfs), len(bigram_idfs), len(trigram_idfs)]
        bars = axes[1, 1].bar(['Unigrams', 'Bigrams', 'Trigrams'], ngram_counts,
                              color=['coral', 'lightblue', 'lightgreen'])
        axes[1, 1].set_ylabel('Number of Features', fontsize=12)
        axes[1, 1].set_title('Feature Distribution by N-gram Type', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height):,}',
                           ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Enhanced visualization saved to: {save_path}")
        plt.close()
        
        # Print statistics
        print(f"\n📊 Vocabulary Statistics:")
        print(f"   Unigrams:  {len(unigram_idfs):,} ({len(unigram_idfs)/self.vocabulary_size*100:.1f}%)")
        print(f"   Bigrams:   {len(bigram_idfs):,} ({len(bigram_idfs)/self.vocabulary_size*100:.1f}%)")
        print(f"   Trigrams:  {len(trigram_idfs):,} ({len(trigram_idfs)/self.vocabulary_size*100:.1f}%)")
        print(f"   Total:     {self.vocabulary_size:,}")
    
    def save(self, filepath):
        """Save vectorizer to disk"""
        joblib.dump(self.vectorizer, filepath)
        print(f"✅ Vectorizer saved to: {filepath}")
    
    def load(self, filepath):
        """Load vectorizer from disk"""
        self.vectorizer = joblib.load(filepath)
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.vocabulary_size = len(self.feature_names)
        self.ngram_range = self.vectorizer.ngram_range
        self.max_features = self.vectorizer.max_features
        
        print(f"✅ Vectorizer loaded from: {filepath}")
        print(f"   Vocabulary size: {self.vocabulary_size:,}")
        print(f"   N-gram range: {self.ngram_range}")
        print(f"   Sublinear TF: {self.vectorizer.sublinear_tf}")


def demonstrate_feature_extraction():
    """
    Demonstrate how enhanced feature extraction works with examples
    """
    print("=" * 80)
    print("ENHANCED FEATURE EXTRACTION DEMONSTRATION")
    print("=" * 80)
    
    # Sample tweets (already cleaned)
    sample_tweets = [
        "love product amaz",
        "hate product terribl",
        "love love love best",
        "disappoint bad qualiti",
        "great recommend everyon",
        "not good bad servic",  # Test bigram/trigram
        "veri happi best purchas ever"  # Test trigram
    ]
    
    print(f"\nSample tweets: {len(sample_tweets)}")
    for i, tweet in enumerate(sample_tweets, 1):
        print(f"  {i}. {tweet}")
    
    # Create enhanced feature extractor
    extractor = FeatureExtractor(
        max_features=50,  # Small for demo
        max_df=1.0,
        min_df=1,
        ngram_range=(1, 3)  # Trigrams!
    )
    
    # Fit and transform
    X = extractor.fit_transform(sample_tweets)
    
    # Show some features
    print(f"\n" + "="*80)
    print("SAMPLE FEATURES EXTRACTED")
    print("="*80)
    
    features = list(extractor.get_feature_names())
    print(f"\nTotal features: {len(features)}")
    print(f"\nSample features (first 20):")
    for i, feature in enumerate(features[:20], 1):
        ngram_type = f"{'unigram' if len(feature.split())==1 else 'bigram' if len(feature.split())==2 else 'trigram'}"
        print(f"  {i:2d}. {feature:30s} ({ngram_type})")
    
    # Show features for each tweet
    print("\n" + "="*80)
    print("TOP FEATURES FOR EACH TWEET")
    print("="*80)
    
    for i, tweet in enumerate(sample_tweets):
        print(f"\n{i+1}. Tweet: \"{tweet}\"")
        features = extractor.get_top_features_for_text(tweet, top_n=5)
        print("   Top features:")
        for feature, score in features:
            ngram = len(feature.split())
            ngram_label = f"({'1-gram' if ngram==1 else '2-gram' if ngram==2 else '3-gram'})"
            print(f"     • {feature:25s} {ngram_label:8s}: {score:.4f}")
    
    # Get statistics
    print("\n" + "="*80)
    print("VOCABULARY STATISTICS")
    print("="*80)
    
    stats = extractor.get_vocabulary_stats()
    print(f"\n  Total features:  {stats['total_features']:,}")
    print(f"  Unigrams:        {stats['unigrams']:,}")
    print(f"  Bigrams:         {stats['bigrams']:,}")
    print(f"  Trigrams:        {stats['trigrams']:,}")
    print(f"\n  IDF Statistics:")
    print(f"    Mean:   {stats['idf_mean']:.4f}")
    print(f"    Std:    {stats['idf_std']:.4f}")
    print(f"    Min:    {stats['idf_min']:.4f}")
    print(f"    Max:    {stats['idf_max']:.4f}")
    
    return extractor, X


def compare_old_vs_new():
    """
    Compare old (v1.0) vs new (v2.0) feature extraction
    """
    print("=" * 80)
    print("COMPARISON: OLD vs NEW FEATURE EXTRACTION")
    print("=" * 80)
    
    sample_tweets = [
        "not good at all",
        "very happy customer",
        "best purchase ever made"
    ]
    
    print("\nTest tweets:")
    for i, tweet in enumerate(sample_tweets, 1):
        print(f"  {i}. {tweet}")
    
    # Old version (v1.0)
    print("\n" + "-"*80)
    print("OLD VERSION (v1.0)")
    print("-"*80)
    old_extractor = FeatureExtractor(
        max_features=20,
        ngram_range=(1, 2),  # Only bigrams
        min_df=1,
        max_df=1.0
    )
    old_extractor.vectorizer.sublinear_tf = False  # Simulate old version
    X_old = old_extractor.fit_transform(sample_tweets)
    
    # New version (v2.0)
    print("\n" + "-"*80)
    print("NEW VERSION (v2.0)")
    print("-"*80)
    new_extractor = FeatureExtractor(
        max_features=30,
        ngram_range=(1, 3),  # Trigrams!
        min_df=1,
        max_df=1.0
    )
    X_new = new_extractor.fit_transform(sample_tweets)
    
    # Comparison
    print("\n" + "="*80)
    print("KEY DIFFERENCES")
    print("="*80)
    print(f"\n{'Feature':<30} {'Old (v1.0)':<15} {'New (v2.0)':<15}")
    print("-"*80)
    print(f"{'N-gram range':<30} {str((1,2)):<15} {str((1,3)):<15}")
    print(f"{'Max features':<30} {'5000':<15} {'10000':<15}")
    print(f"{'Sublinear TF':<30} {'False':<15} {'True':<15}")
    print(f"{'Max DF':<30} {'0.9':<15} {'0.8':<15}")
    print(f"{'Min DF':<30} {'5':<15} {'3':<15}")
    
    print("\n" + "="*80)
    print("CAPTURED PHRASES")
    print("="*80)
    
    old_features = set(old_extractor.get_feature_names())
    new_features = set(new_extractor.get_feature_names())
    
    new_only = new_features - old_features
    
    print(f"\n✨ New features captured (not in old version):")
    for feature in sorted(new_only):
        if len(feature.split()) == 3:
            print(f"   • {feature} (trigram)")
    
    print(f"\n📊 Summary:")
    print(f"   Old version features: {len(old_features)}")
    print(f"   New version features: {len(new_features)}")
    print(f"   Additional features:  {len(new_only)}")


if __name__ == "__main__":
    print("\n🚀 Running demonstrations...\n")
    
    # Demo 1: Basic demonstration
    demonstrate_feature_extraction()
    
    print("\n\n")
    
    # Demo 2: Comparison
    compare_old_vs_new()
    
    print("\n\n✅ Demonstrations complete!")