"""
Configuration settings for Business Mapper API v2.0
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""
    
    # Flask Configuration
    FLASK_ENV = os.getenv('FLASK_ENV', 'production')
    SECRET_KEY = os.getenv('SECRET_KEY', os.urandom(32).hex())
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 8080))
    
    # API Keys
    SERPER_API_KEY = os.getenv('SERPER_API_KEY')
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    ADMIN_KEY = os.getenv('ADMIN_KEY')
    
    # Database
    DATABASE_PATH = os.getenv('DATABASE_PATH', './data/mapper.db')
    
    # Geocoding
    GEOCODING_SERVICE = os.getenv('GEOCODING_SERVICE', 'open-meteo')
    GEOCODING_CACHE_TTL = int(os.getenv('GEOCODING_CACHE_TTL', 86400))
    
    # Search Settings
    DEFAULT_ZOOM = int(os.getenv('DEFAULT_ZOOM', 14))
    DEFAULT_MAX_PAGES = int(os.getenv('DEFAULT_MAX_PAGES', 3))
    MAX_CONCURRENT_REQUESTS = int(os.getenv('MAX_CONCURRENT_REQUESTS', 5))
    
    # Cost Settings
    SEARCH_COST_SAFETY_THRESHOLD_USD = float(os.getenv('SEARCH_COST_SAFETY_THRESHOLD_USD', 1.0))
    CREDITS_PER_PAGE = 3
    COST_PER_CREDIT = 0.001
    
    # AI Settings
    AI_VALIDATION_ENABLED = os.getenv('AI_VALIDATION_ENABLED', 'True').lower() == 'true'
    AI_VALIDATION_MODEL = os.getenv('AI_VALIDATION_MODEL', 'llama-3.1-8b-instant')
    AI_VALIDATION_TEMPERATURE = float(os.getenv('AI_VALIDATION_TEMPERATURE', 0.1))
    AI_VALIDATION_MAX_TOKENS = int(os.getenv('AI_VALIDATION_MAX_TOKENS', 10))
    
    # Cache Settings
    VALIDATION_CACHE_ENABLED = os.getenv('VALIDATION_CACHE_ENABLED', 'True').lower() == 'true'
    VALIDATION_CACHE_TTL_DAYS = int(os.getenv('VALIDATION_CACHE_TTL_DAYS', 30))
    
    # City Classification
    AI_CITY_CLASSIFICATION_ENABLED = os.getenv('AI_CITY_CLASSIFICATION_ENABLED', 'True').lower() == 'true'
    AI_CITY_CLASSIFICATION_TTL_DAYS = int(os.getenv('AI_CITY_CLASSIFICATION_TTL_DAYS', 365))
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    @classmethod
    def validate(cls):
        """Validate required settings."""
        errors = []
        
        if not cls.SERPER_API_KEY:
            errors.append("SERPER_API_KEY is required")
        
        if not cls.GROQ_API_KEY:
            errors.append("GROQ_API_KEY is required for AI-powered query expansion")
        
        # ADMIN_KEY is optional - only used for admin endpoints
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")


# Create settings instance
settings = Settings()