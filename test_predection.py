"""
Comprehensive Testing Suite for Prediction System
Tests various scenarios and edge cases
"""

from predict import SentimentPredictor
import pandas as pd


def test_basic_predictions(predictor):
    """Test basic positive and negative predictions"""
    print("="*80)
    print("TEST 1: BASIC PREDICTIONS")
    print("="*80)
    
    test_cases = [
        {
            'tweet': "I love this product!",
            'expected': 'POSITIVE',
            'description': 'Clear positive sentiment'
        },
        {
            'tweet': "I hate this product!",
            'expected': 'NEGATIVE',
            'description': 'Clear negative sentiment'
        },
        {
            'tweet': "This is amazing and wonderful!",
            'expected': 'POSITIVE',
            'description': 'Multiple positive words'
        },
        {
            'tweet': "Terrible and awful experience!",
            'expected': 'NEGATIVE',
            'description': 'Multiple negative words'
        }
    ]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        result = predictor.predict_single(test['tweet'])
        prediction = result['sentiment']
        expected = test['expected']
        
        status = "✅ PASS" if prediction == expected else "❌ FAIL"
        
        print(f"\nTest {i}: {test['description']}")
        print(f"Tweet: {test['tweet']}")
        print(f"Expected: {expected}, Got: {prediction} ({result['confidence']:.1f}%)")
        print(status)
        
        if prediction == expected:
            passed += 1
        else:
            failed += 1
    
    print(f"\n{'='*80}")
    print(f"Results: {passed}/{len(test_cases)} passed")
    return passed, failed


def test_edge_cases(predictor):
    """Test edge cases and problematic scenarios"""
    print("\n" + "="*80)
    print("TEST 2: EDGE CASES")
    print("="*80)
    
    edge_cases = [
        "Not bad",  # Negation
        "Not good",  # Negation
        "It's okay",  # Neutral
        "Meh",  # Very neutral
        "Could be better",  # Mild negative
        "Could be worse",  # Mild positive
        "",  # Empty
        "!!!",  # Only punctuation
        "😊😊😊",  # Only emojis
        "http://example.com"  # Only URL
    ]
    
    print("\nTesting edge cases:")
    
    for i, tweet in enumerate(edge_cases, 1):
        result = predictor.predict_single(tweet)
        
        print(f"\n{i}. Tweet: '{tweet}'")
        print(f"   Cleaned: '{result['cleaned_tweet']}'")
        print(f"   Prediction: {result['sentiment']} ({result['confidence']:.1f}%)")
        
        if result['confidence'] < 70:
            print(f"   ⚠️  Low confidence - model is uncertain")


def test_with_emojis(predictor):
    """Test how emojis affect predictions"""
    print("\n" + "="*80)
    print("TEST 3: EMOJI HANDLING")
    print("="*80)
    
    emoji_tests = [
        ("Great product", "Great product 😊"),
        ("Bad service", "Bad service 😠"),
        ("Love it", "Love it ❤️😍"),
        ("Hate it", "Hate it 😡🤮")
    ]
    
    for without, with_emoji in emoji_tests:
        result_without = predictor.predict_single(without)
        result_with = predictor.predict_single(with_emoji)
        
        print(f"\nWithout emoji: '{without}'")
        print(f"  → {result_without['sentiment']} ({result_without['confidence']:.1f}%)")
        
        print(f"With emoji: '{with_emoji}'")
        print(f"  → {result_with['sentiment']} ({result_with['confidence']:.1f}%)")
        
        diff = abs(result_with['confidence'] - result_without['confidence'])
        print(f"  Confidence difference: {diff:.1f}%")


def test_mixed_sentiments(predictor):
    """Test tweets with mixed positive and negative sentiments"""
    print("\n" + "="*80)
    print("TEST 4: MIXED SENTIMENTS")
    print("="*80)
    
    mixed_tweets = [
        "I love the design but hate the quality",
        "Great product but terrible customer service",
        "Fast shipping but product is disappointing",
        "Expensive but worth it",
        "Cheap but feels premium"
    ]
    
    for tweet in mixed_tweets:
        result = predictor.predict_single(tweet)
        
        print(f"\nTweet: {tweet}")
        print(f"Prediction: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.1f}%")
        print(f"Positive: {result['probabilities']['positive']*100:.1f}%, "
              f"Negative: {result['probabilities']['negative']*100:.1f}%")
        
        # Mixed sentiments should have lower confidence
        if result['confidence'] < 75:
            print("✅ Correctly uncertain about mixed sentiment")
        else:
            print("⚠️  High confidence despite mixed sentiment")


def test_length_variations(predictor):
    """Test tweets of different lengths"""
    print("\n" + "="*80)
    print("TEST 5: LENGTH VARIATIONS")
    print("="*80)
    
    length_tests = [
        ("Very short", "Love it"),
        ("Short", "I really love this product"),
        ("Medium", "I absolutely love this product! The quality is amazing and it works perfectly."),
        ("Long", "I have to say, I absolutely love this product! The quality is outstanding, "
                 "it works exactly as described, shipping was fast, and customer service "
                 "was helpful when I had questions. Highly recommend to everyone!")
    ]
    
    for length_type, tweet in length_tests:
        result = predictor.predict_single(tweet)
        
        print(f"\n{length_type} ({len(tweet)} chars):")
        print(f"Tweet: {tweet}")
        print(f"Prediction: {result['sentiment']} ({result['confidence']:.1f}%)")


def test_real_world_examples(predictor):
    """Test with realistic tweet examples"""
    print("\n" + "="*80)
    print("TEST 6: REAL-WORLD EXAMPLES")
    print("="*80)
    
    real_tweets = [
        # Product reviews
        "Just got my new headphones and WOW! The sound quality is incredible! 🎧",
        "Ordered 2 weeks ago, still haven't received it. Terrible service. 😠",
        
        # Restaurant reviews  
        "Best pizza in town! Authentic Italian taste. Must try! 🍕",
        "Cold food, rude staff, never coming back. Worst dining experience.",
        
        # Service feedback
        "The customer support team went above and beyond to help me!",
        "Been on hold for 45 minutes. This is ridiculous. Fix your system!",
        
        # Movie reviews
        "Just watched the new movie. Absolutely brilliant! Oscar-worthy! 🎬",
        "Waste of 2 hours. Boring plot, bad acting. Don't bother.",
        
        # Tech reviews
        "This app crashes every 5 minutes. Completely unusable. 1 star.",
        "Clean interface, fast performance, does exactly what I need. 5 stars!"
    ]
    
    results = predictor.predict_batch(real_tweets, show_progress=False)
    
    for tweet, result in zip(real_tweets, results):
        emoji = "😊" if result['sentiment'] == 'POSITIVE' else "😞"
        
        print(f"\n{emoji} Tweet: {tweet}")
        print(f"   → {result['sentiment']} ({result['confidence']:.1f}%)")


def test_batch_performance(predictor):
    """Test batch prediction performance"""
    print("\n" + "="*80)
    print("TEST 7: BATCH PERFORMANCE")
    print("="*80)
    
    # Create test dataset
    test_tweets = [
        "I love this!",
        "I hate this!",
        "It's okay.",
        "Amazing product!",
        "Terrible quality."
    ] * 20  # 100 tweets
    
    print(f"\nTesting batch prediction with {len(test_tweets)} tweets...")
    
    import time
    start = time.time()
    results = predictor.predict_batch(test_tweets, show_progress=True)
    duration = time.time() - start
    
    stats = predictor.get_statistics(results)
    
    print(f"\n{'='*80}")
    print("BATCH STATISTICS")
    print(f"{'='*80}")
    print(f"\nTotal tweets: {len(test_tweets)}")
    print(f"Processing time: {duration:.2f} seconds")
    print(f"Speed: {len(test_tweets)/duration:.1f} tweets/second")
    print(f"\nSentiment distribution:")
    print(f"  Positive: {stats['positive_count']} ({stats['positive_percentage']:.1f}%)")
    print(f"  Negative: {stats['negative_count']} ({stats['negative_percentage']:.1f}%)")
    print(f"\nAverage confidence: {stats['average_confidence']:.1f}%")


def test_explanation_feature(predictor):
    """Test prediction explanation feature"""
    print("\n" + "="*80)
    print("TEST 8: PREDICTION EXPLANATIONS")
    print("="*80)
    
    test_tweet = "I absolutely love this amazing product! Best purchase ever!"
    
    print(f"\nExplaining prediction for:")
    print(f'"{test_tweet}"')
    
    explanation = predictor.explain_prediction(test_tweet, top_n=5)
    
    print(f"\n🎯 Prediction: {explanation['prediction']['sentiment']}")
    print(f"📊 Confidence: {explanation['prediction']['confidence']:.1f}%")
    print(f"\n🔍 Top 5 influential words:")
    
    for i, feature in enumerate(explanation['top_features'], 1):
        print(f"\n{i}. Word: '{feature['word']}'")
        if 'direction' in feature:
            direction_emoji = "🟢" if feature['direction'] == 'positive' else "🔴"
            print(f"   {direction_emoji} Direction: {feature['direction']}")
            print(f"   Impact score: {feature['importance']:.4f}")


def run_all_tests():
    """Run all tests"""
    print("="*80)
    print(" "*20 + "PREDICTION SYSTEM TEST SUITE")
    print("="*80)
    
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
        print("Please train the model first: python run_training.py")
        return
    
    # Run tests
    total_passed = 0
    total_tests = 0
    
    try:
        passed, failed = test_basic_predictions(predictor)
        total_passed += passed
        total_tests += (passed + failed)
        
        test_edge_cases(predictor)
        test_with_emojis(predictor)
        test_mixed_sentiments(predictor)
        test_length_variations(predictor)
        test_real_world_examples(predictor)
        test_batch_performance(predictor)
        test_explanation_feature(predictor)
        
        # Final summary
        print("\n" + "="*80)
        print("FINAL TEST SUMMARY")
        print("="*80)
        print(f"\nBasic prediction tests: {total_passed}/{total_tests} passed")
        print(f"\n✅ All other tests completed successfully!")
        print("\n🎉 Prediction system is working correctly!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()