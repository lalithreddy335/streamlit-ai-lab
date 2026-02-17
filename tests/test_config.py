"""Tests for configuration"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import Config


class TestConfig:
    """Test configuration management"""
    
    def test_config_app_name(self):
        assert Config.APP_NAME == "Streamlit AI Lab"
    
    def test_config_version(self):
        assert hasattr(Config, "VERSION")
        assert Config.VERSION == "1.0.0"
    
    def test_get_dict(self):
        config_dict = Config.get_dict()
        assert isinstance(config_dict, dict)
        assert "APP_NAME" in config_dict
        assert "VERSION" in config_dict
    
    def test_validate(self):
        # Should not raise exception
        assert Config.validate() is True
