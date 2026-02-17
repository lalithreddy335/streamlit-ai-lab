"""
Utility functions for Streamlit AI Lab
"""

import logging
from typing import Any, Dict, List, Optional
from functools import wraps
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def timer(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} executed in {elapsed:.4f} seconds")
        return result
    return wrapper


def validate_input(data: Any, expected_type: type) -> bool:
    """
    Validate input data type
    
    Args:
        data: Data to validate
        expected_type: Expected type
    
    Returns:
        True if valid, False otherwise
    """
    return isinstance(data, expected_type)


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """
    Format timestamp to readable string
    
    Args:
        dt: DateTime object, defaults to now
    
    Returns:
        Formatted timestamp string
    """
    dt = dt or datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def safe_get(dictionary: Dict, key: str, default: Any = None) -> Any:
    """
    Safely get dictionary value with default
    
    Args:
        dictionary: Dictionary to access
        key: Key to retrieve
        default: Default value if key not found
    
    Returns:
        Value or default
    """
    return dictionary.get(key, default)


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """
    Split list into chunks
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
    
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def retry(max_attempts: int = 3, delay: float = 1.0):
    """
    Decorator for retry logic
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Delay between attempts in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        logger.error(f"Failed after {max_attempts} attempts: {str(e)}")
                        raise
                    logger.warning(f"Attempt {attempts} failed, retrying in {delay}s...")
                    time.sleep(delay)
        return wrapper
    return decorator


def merge_dicts(*dicts: Dict) -> Dict:
    """
    Merge multiple dictionaries
    
    Args:
        *dicts: Dictionaries to merge
    
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result


class PerformanceTracker:
    """Track performance metrics"""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
    
    def record(self, metric_name: str, value: float):
        """Record a metric"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
    
    def get_average(self, metric_name: str) -> float:
        """Get average value for metric"""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return 0.0
        return sum(self.metrics[metric_name]) / len(self.metrics[metric_name])
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all metrics"""
        return {
            metric: {
                "avg": self.get_average(metric),
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }
            for metric, values in self.metrics.items()
        }
