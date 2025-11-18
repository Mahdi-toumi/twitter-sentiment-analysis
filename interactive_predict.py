"""
Interactive Prediction Script
Test your trained model with custom tweets
"""

from predict import SentimentPredictor
import sys


def print_header():
    """Print application header"""
    print("\n" + "="*80)
    print(" "*20 + "🎭 TWITTER SENTIMENT ANALYZER 🎭")
    print("="*80)


def print_result(result):
    """Pretty print prediction result"""
    sentiment = result['sentiment']
    confidence = result['confidence']
    
    # Choose emoji based on sentiment
    emoji = "😊" if sentiment == 'POSITIVE' else "😞"
    
    # Choose color indicator
    indicator = "🟢" if sentiment == 'POSITIVE' else "🔴"
    
    print("\n" + "="*80)
    print("PREDICTION RESULT")
    print("="*80)
    
    print(f"\n📝 Original Tweet:")
    print(f"   {result['original_tweet']}")
    
    print(f"\n🧹 Cleaned Tweet:")
    print(f"   {result['cleaned_tweet']}")
    
    print(f"\n{indicator} Sentiment: {sentiment} {emoji}")
    
    # Confidence bar
    bar_length = 50
    filled = int((confidence / 100) * bar_length)
    bar = "█" * filled + "░" * (bar_length - filled)
    
    print(f"\n📊 Confidence: {confidence:.1f}%")
    print(f"   {bar}")
    
    print(f"\n📈 Detailed Probabilities:")
    print(f"   Positive: {result['probabilities']['positive']*100:>5.1f}%")
    print(f"   Negative: {result['probabilities']['negative']*100:>5.1f}%")
    
    # Interpretation
    print(f"\n💡 Interpretation:")
    if confidence >= 90:
        print(f"   The model is VERY CONFIDENT about this prediction.")
    elif confidence >= 75:
        print(f"   The model is CONFIDENT about this prediction.")
    elif confidence >= 60:
        print(f"   The model is MODERATELY CONFIDENT about this prediction.")
    else:
        print(f"   The model is UNCERTAIN about this prediction.")


def interactive_mode(predictor):
    """Run interactive prediction mode"""
    print_header()
    print("\n🎯 Interactive Mode - Enter tweets to analyze their sentiment")
    print("💡 Type 'quit' or 'exit' to stop")
    print("💡 Type 'examples' to see sample predictions")
    print("💡 Type 'help' for more options")
    
    while True:
        print("\n" + "-"*80)
        tweet = input("\n✍️  Enter a tweet: ").strip()
        
        if not tweet:
            print("⚠️  Please enter a tweet!")
            continue
        
        # Commands
        if tweet.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Thank you for using the Sentiment Analyzer!")
            break
        
        elif tweet.lower() == 'examples':
            show_examples(predictor)
            continue
        
        elif tweet.lower() == 'help':
            show_help()
            continue
        
        elif tweet.lower() == 'batch':
            batch_mode(predictor)
            continue
        
        # Make prediction
        try:
            result = predictor.predict_single(tweet)
            print_result(result)
        except Exception as e:
            print(f"\n❌ Error making prediction: {e}")


def show_examples(predictor):
    """Show example predictions"""
    examples = [
        "I love this product! Best purchase ever! 😍",
        "Terrible service. Very disappointed. 😠",
        "It's okay, nothing special.",
        "Absolutely amazing! Highly recommend!",
        "Worst experience ever! Never again!"
    ]
    
    print("\n" + "="*80)
    print("EXAMPLE PREDICTIONS")
    print("="*80)
    
    for i, tweet in enumerate(examples, 1):
        print(f"\n{'─'*80}")
        print(f"Example {i}:")
        result = predictor.predict_single(tweet)
        print_result(result)


def batch_mode(predictor):
    """Batch prediction mode"""
    print("\n" + "="*80)
    print("BATCH PREDICTION MODE")
    print("="*80)
    print("\nEnter multiple tweets (one per line)")
    print("Type 'done' when finished")
    
    tweets = []
    while True:
        tweet = input(f"\nTweet {len(tweets) + 1}: ").strip()
        
        if tweet.lower() == 'done':
            break
        
        if tweet:
            tweets.append(tweet)
    
    if not tweets:
        print("⚠️  No tweets entered!")
        return
    
    print(f"\n📊 Analyzing {len(tweets)} tweets...\n")
    
    results = predictor.predict_batch(tweets, show_progress=False)
    stats = predictor.get_statistics(results)
    
    # Show individual results
    for i, (tweet, result) in enumerate(zip(tweets, results), 1):
        print(f"\n{'─'*80}")
        print(f"Tweet {i}: {tweet}")
        sentiment_emoji = "😊" if result['sentiment'] == 'POSITIVE' else "😞"
        print(f"→ {result['sentiment']} {sentiment_emoji} ({result['confidence']:.1f}%)")
    
    # Show statistics
    print("\n" + "="*80)
    print("BATCH STATISTICS")
    print("="*80)
    print(f"\nTotal Tweets: {stats['total_tweets']}")
    print(f"\n📊 Sentiment Distribution:")
    print(f"   😊 Positive: {stats['positive_count']} ({stats['positive_percentage']:.1f}%)")
    print(f"   😞 Negative: {stats['negative_count']} ({stats['negative_percentage']:.1f}%)")
    print(f"\n📈 Average Confidence: {stats['average_confidence']:.1f}%")


def show_help():
    """Show help information"""
    print("\n" + "="*80)
    print("HELP - AVAILABLE COMMANDS")
    print("="*80)
    print("""
Commands:
  - Just type any tweet to analyze it
  - 'examples'  : See example predictions
  - 'batch'     : Analyze multiple tweets at once
  - 'help'      : Show this help message
  - 'quit/exit' : Exit the program

Tips:
  - Use emojis, they add context! 😊 😞
  - The model works best with clear emotional language
  - Neutral tweets may have lower confidence scores
  - Check the confidence level - higher means more certain
    """)


def file_mode(predictor, filepath):
    """Predict sentiments for tweets in a file"""
    print(f"\n📂 Loading tweets from: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tweets = [line.strip() for line in f if line.strip()]
        
        print(f"✅ Loaded {len(tweets)} tweets")
        
        results = predictor.predict_batch(tweets, show_progress=True)
        stats = predictor.get_statistics(results)
        
        # Save results
        output_file = filepath.replace('.txt', '_predictions.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Tweet\tSentiment\tConfidence\n")
            for tweet, result in zip(tweets, results):
                f.write(f"{tweet}\t{result['sentiment']}\t{result['confidence']:.1f}%\n")
        
        print(f"\n✅ Results saved to: {output_file}")
        
        # Show statistics
        print("\n" + "="*80)
        print("STATISTICS")
        print("="*80)
        print(f"\nTotal Tweets: {stats['total_tweets']}")
        print(f"\n📊 Sentiment Distribution:")
        print(f"   Positive: {stats['positive_count']} ({stats['positive_percentage']:.1f}%)")
        print(f"   Negative: {stats['negative_count']} ({stats['negative_percentage']:.1f}%)")
        print(f"\n📈 Average Confidence: {stats['average_confidence']:.1f}%")
        
    except FileNotFoundError:
        print(f"❌ Error: File not found: {filepath}")
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Main function"""
    # Initialize predictor
    predictor = SentimentPredictor()
    
    # Load model
    try:
        predictor.load_model(
            model_path='models/logistic_regression_model.pkl',
            vectorizer_path='models/tfidf_vectorizer.pkl'
        )
    except FileNotFoundError:
        print("\n❌ Error: Model files not found!")
        print("\nPlease train the model first:")
        print("  python run_training.py")
        return
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        return
    
    # Check command line arguments
    if len(sys.argv) > 1:
        # File mode
        filepath = sys.argv[1]
        file_mode(predictor, filepath)
    else:
        # Interactive mode
        interactive_mode(predictor)


if __name__ == "__main__":
    main()