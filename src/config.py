"""
Configuration management for Streamlit AI Lab
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables
load_dotenv()


class Config:
    """Application configuration"""
    
    # Application settings
    APP_NAME = "Streamlit AI Lab"
    VERSION = "1.0.0"
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # API Keys and Credentials
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
    
    # Model Settings
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    TOP_P = float(os.getenv("TOP_P", "0.9"))
    
    # Database Settings
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///app.db")
    DATABASE_ECHO = DEBUG
    
    # Cache Settings
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
    ENABLE_CACHE = os.getenv("ENABLE_CACHE", "True").lower() == "true"
    
    # Feature Flags
    ENABLE_FILE_UPLOAD = os.getenv("ENABLE_FILE_UPLOAD", "True").lower() == "true"
    ENABLE_CHAT = os.getenv("ENABLE_CHAT", "True").lower() == "true"
    ENABLE_ANALYSIS = os.getenv("ENABLE_ANALYSIS", "True").lower() == "true"
    
    # Rate Limiting
    RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "True").lower() == "true"
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_PERIOD = int(os.getenv("RATE_LIMIT_PERIOD", "60"))
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "app.log")
    
    # File Upload Settings
    MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", "104857600"))  # 100MB
    ALLOWED_FILE_TYPES = ["csv", "json", "txt", "xlsx"]
    UPLOAD_DIRECTORY = os.getenv("UPLOAD_DIRECTORY", "/tmp/uploads")
    
    @classmethod
    def get_dict(cls) -> Dict[str, Any]:
        """Get all configuration as dictionary"""
        return {
            key: getattr(cls, key)
            for key in dir(cls)
            if not key.startswith('_') and key.isupper()
        }
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validate critical configuration settings
        
        Returns:
            True if configuration is valid
        """
        if not cls.DATABASE_URL:
            raise ValueError("DATABASE_URL not set")
        
        return True


# Create singleton instance
config = Config()
