"""
Simple Training Script
Run this to train your sentiment analysis model
"""

import os
import sys

def check_requirements():
    """Check if all required packages are installed"""
    required = ['pandas', 'numpy', 'sklearn', 'nltk', 'matplotlib', 'joblib']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("\nInstall them with:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    print("✅ All required packages installed")
    return True


def check_data():
    """Check if data file exists"""
    data_paths = [
        'data/sentiment140_cleaned.csv',
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            print(f"✅ Found data file: {path}")
            return path
    
    print("❌ Data file not found!")
    print("\nExpected locations:")
    for path in data_paths:
        print(f"  - {path}")
    print("\nPlease download the dataset first.")
    return None


def create_directories():
    """Create necessary directories"""
    dirs = ['models', 'notebooks', 'data']
    
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"✅ Created directory: {dir_name}")
        else:
            print(f"✅ Directory exists: {dir_name}")


def main():
    """Main training pipeline"""
    print("="*80)
    print("TWITTER SENTIMENT ANALYSIS - TRAINING SCRIPT")
    print("="*80)
    
    # Step 1: Check requirements
    print("\n" + "="*80)
    print("STEP 1: Checking Requirements")
    print("="*80)
    if not check_requirements():
        return
    
    # Step 2: Create directories
    print("\n" + "="*80)
    print("STEP 2: Creating Directories")
    print("="*80)
    create_directories()
    
    # Step 3: Check data
    print("\n" + "="*80)
    print("STEP 3: Checking Data")
    print("="*80)
    data_path = check_data()
    if not data_path:
        return
    
    # Step 4: Import training module
    print("\n" + "="*80)
    print("STEP 4: Loading Training Module")
    print("="*80)
    try:
        from train_model import train_full_pipeline
        print("✅ Training module loaded")
    except ImportError as e:
        print(f"❌ Error importing training module: {e}")
        print("\nMake sure train_model.py is in the same directory")
        return
    
    # Step 5: Train models
    print("\n" + "="*80)
    print("STEP 5: Starting Training")
    print("="*80)
    
    # Ask user about sample size
    print("\nHow many samples to use for training?")
    print("1. Small (10,000 samples) - Fast, for testing (~30 seconds)")
    print("2. Medium (100,000 samples) - Balanced (~5 minutes)")
    print("3. Large (500,000 samples) - Better accuracy (~20 minutes)")
    print("4. Full dataset (1,600,000 samples) - Best accuracy (~1 hour)")
    
    choice = input("\nEnter your choice (1-4) [default: 2]: ").strip() or "2"
    
    sample_sizes = {
        '1': 10000,
        '2': 100000,
        '3': 500000,
        '4': None
    }
    
    sample_size = sample_sizes.get(choice, 100000)
    
    if sample_size:
        print(f"\n✅ Training with {sample_size:,} samples")
    else:
        print(f"\n✅ Training with full dataset")
    
    # Run training
    try:
        nb_model, lr_model, extractor, results = train_full_pipeline(
            data_path=data_path,
            sample_size=sample_size,
            test_size=0.2,
            use_hyperparameter_tuning=True,  # Set True for tuning (adds 5-60 min)
            use_ensemble=True,  # Train ensemble models
            quick_tuning=False  # Quick tuning if enabled
        )
        
        print("\n" + "="*80)
        print("🎉 TRAINING COMPLETE!")
        print("="*80)
        
        print("\n📊 Final Results:")
        print("-"*80)
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name:12s}: {value:.4f}")
        
        print("\n" + "="*80)
        print("Next Steps:")
        print("="*80)
        print("1. Check models/ directory for saved models")
        print("2. Review performance visualizations (PNG files)")
        print("3. Test predictions with app.py")
        print("4. Deploy to production!")
        
    except KeyboardInterrupt:
        print("\n\n❌ Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()