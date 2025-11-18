"""
Configuration Module
Centralized configuration for Flask application
"""

import os
from datetime import timedelta


class Config:
    """Base configuration class"""
    
    # Flask Settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = False
    TESTING = False
    
    # Application Settings
    APP_NAME = "Twitter Sentiment Analyzer"
    APP_VERSION = "1.0.0"
    APP_DESCRIPTION = "Sentiment analysis for tweets"
    
    # Model Paths
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    MODEL_PATH = os.path.join(MODEL_DIR, 'logistic_regression_model.pkl')
    VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
    
    # Upload Settings (for batch predictions)
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'txt', 'csv'}
    
    # Session Settings
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    SESSION_COOKIE_SECURE = False  # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Rate Limiting (for API)
    RATELIMIT_ENABLED = True
    RATELIMIT_DEFAULT = "100 per hour"
    RATELIMIT_STORAGE_URL = "memory://"
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Prediction Settings
    MAX_TWEET_LENGTH = 280  # Twitter's limit
    BATCH_SIZE_LIMIT = 1000  # Max tweets per batch
    CONFIDENCE_THRESHOLD_LOW = 0.60
    CONFIDENCE_THRESHOLD_MEDIUM = 0.75
    CONFIDENCE_THRESHOLD_HIGH = 0.90
    
    # API Settings
    API_VERSION = "v1"
    API_TITLE = "Sentiment Analysis API"
    API_DESCRIPTION = "REST API for Twitter sentiment analysis"
    
    # Cache Settings (for future optimization)
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 300  # 5 minutes


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False
    
    # More verbose logging in development
    LOG_LEVEL = 'DEBUG'
    
    # Disable rate limiting in development
    RATELIMIT_ENABLED = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    
    # Use environment variables in production
    SECRET_KEY = os.environ.get('SECRET_KEY')
    
    if not SECRET_KEY:
        raise ValueError("SECRET_KEY environment variable must be set in production!")
    
    # Enable secure cookies in production
    SESSION_COOKIE_SECURE = True
    
    # Stricter rate limiting
    RATELIMIT_DEFAULT = "50 per hour"
    
    # Production logging
    LOG_LEVEL = 'WARNING'


class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = False
    TESTING = True
    
    # Use test models (if different)
    MODEL_PATH = os.path.join(Config.MODEL_DIR, 'test_model.pkl')
    
    # Disable rate limiting for tests
    RATELIMIT_ENABLED = False
    
    # Short session lifetime for tests
    PERMANENT_SESSION_LIFETIME = timedelta(minutes=5)


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(config_name=None):
    """
    Get configuration object based on environment
    
    Parameters:
    -----------
    config_name : str, optional
        Name of configuration ('development', 'production', 'testing')
        If None, uses FLASK_ENV environment variable
    
    Returns:
    --------
    Config object
    """
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')
    
    return config.get(config_name, DevelopmentConfig)


# Helper functions
def ensure_directories_exist():
    """Create necessary directories if they don't exist"""
    directories = [
        Config.MODEL_DIR,
        Config.UPLOAD_FOLDER,
        os.path.join(Config.BASE_DIR, 'logs')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ All required directories created")


def validate_config():
    """Validate that required configuration is set correctly"""
    config = get_config()
    
    issues = []
    
    # Check if model files exist
    if not os.path.exists(config.MODEL_PATH):
        issues.append(f"Model file not found: {config.MODEL_PATH}")
    
    if not os.path.exists(config.VECTORIZER_PATH):
        issues.append(f"Vectorizer file not found: {config.VECTORIZER_PATH}")
    
    # Check secret key in production
    if config.DEBUG is False and config.SECRET_KEY == 'dev-secret-key-change-in-production':
        issues.append("SECRET_KEY must be changed in production!")
    
    if issues:
        print("⚠️  Configuration Issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    print("✅ Configuration validated successfully")
    return True


if __name__ == "__main__":
    # Test configuration
    print("="*80)
    print("CONFIGURATION TEST")
    print("="*80)
    
    # Test different environments
    for env in ['development', 'production', 'testing']:
        print(f"\n{env.upper()} Configuration:")
        conf = config[env]()
        print(f"  DEBUG: {conf.DEBUG}")
        print(f"  TESTING: {conf.TESTING}")
        print(f"  LOG_LEVEL: {conf.LOG_LEVEL}")
        print(f"  RATELIMIT_ENABLED: {conf.RATELIMIT_ENABLED}")
    
    # Ensure directories
    print("\n" + "="*80)
    ensure_directories_exist()
    
    # Validate
    print("\n" + "="*80)
    validate_config()