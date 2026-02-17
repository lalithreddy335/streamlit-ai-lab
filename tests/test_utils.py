"""Tests for utility functions"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import (
    validate_input,
    format_timestamp,
    safe_get,
    chunk_list,
    merge_dicts,
)


class TestValidateInput:
    """Test input validation"""
    
    def test_valid_string(self):
        assert validate_input("hello", str) is True
    
    def test_invalid_string(self):
        assert validate_input(123, str) is False
    
    def test_valid_int(self):
        assert validate_input(42, int) is True


class TestFormatTimestamp:
    """Test timestamp formatting"""
    
    def test_format_timestamp(self):
        result = format_timestamp()
        assert isinstance(result, str)
        assert len(result) == 19  # "YYYY-MM-DD HH:MM:SS"


class TestSafeGet:
    """Test safe dictionary access"""
    
    def test_get_existing_key(self):
        d = {"key": "value"}
        assert safe_get(d, "key") == "value"
    
    def test_get_missing_key(self):
        d = {"key": "value"}
        assert safe_get(d, "missing") is None
    
    def test_get_with_default(self):
        d = {}
        assert safe_get(d, "key", "default") == "default"


class TestChunkList:
    """Test list chunking"""
    
    def test_chunk_list(self):
        lst = [1, 2, 3, 4, 5]
        result = chunk_list(lst, 2)
        assert result == [[1, 2], [3, 4], [5]]
    
    def test_chunk_list_exact(self):
        lst = [1, 2, 3, 4]
        result = chunk_list(lst, 2)
        assert result == [[1, 2], [3, 4]]


class TestMergeDicts:
    """Test dictionary merging"""
    
    def test_merge_two_dicts(self):
        d1 = {"a": 1}
        d2 = {"b": 2}
        result = merge_dicts(d1, d2)
        assert result == {"a": 1, "b": 2}
    
    def test_merge_overlapping_dicts(self):
        d1 = {"a": 1, "b": 2}
        d2 = {"b": 3, "c": 4}
        result = merge_dicts(d1, d2)
        assert result == {"a": 1, "b": 3, "c": 4}
