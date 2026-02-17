"""
Data Processing Module
Handle data loading, processing, and transformation
"""

from typing import Optional, Dict, Any, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DataProcessor:
    """Process and transform data"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @staticmethod
    def load_csv(filepath: str) -> Optional[Any]:
        """
        Load CSV file
        
        Args:
            filepath: Path to CSV file
        
        Returns:
            Loaded dataframe or None
        """
        try:
            import pandas as pd
            df = pd.read_csv(filepath)
            logger.info(f"Loaded CSV: {filepath} ({len(df)} rows)")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            return None
    
    @staticmethod
    def load_json(filepath: str) -> Optional[Dict]:
        """
        Load JSON file
        
        Args:
            filepath: Path to JSON file
        
        Returns:
            Loaded data or None
        """
        try:
            import json
            with open(filepath, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded JSON: {filepath}")
            return data
        except Exception as e:
            logger.error(f"Error loading JSON: {str(e)}")
            return None
    
    @staticmethod
    def clean_data(df: Any) -> Any:
        """
        Clean dataframe
        
        Args:
            df: Input dataframe
        
        Returns:
            Cleaned dataframe
        """
        try:
            # Remove duplicates
            df = df.drop_duplicates()
            
            # Remove rows with all NaN
            df = df.dropna(how='all')
            
            # Fill remaining NaN with appropriate values
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna('Unknown')
                else:
                    df[col] = df[col].fillna(df[col].mean())
            
            logger.info("Data cleaned successfully")
            return df
        
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            return None
    
    @staticmethod
    def normalize_data(df: Any) -> Any:
        """
        Normalize dataframe columns
        
        Args:
            df: Input dataframe
        
        Returns:
            Normalized dataframe
        """
        try:
            from sklearn.preprocessing import StandardScaler
            
            numeric_cols = df.select_dtypes(include=['number']).columns
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            
            logger.info("Data normalized successfully")
            return df
        
        except Exception as e:
            logger.error(f"Error normalizing data: {str(e)}")
            return None
    
    @staticmethod
    def aggregate_data(df: Any, group_by: List[str], agg_dict: Dict[str, str]) -> Any:
        """
        Aggregate data by columns
        
        Args:
            df: Input dataframe
            group_by: Columns to group by
            agg_dict: Aggregation functions
        
        Returns:
            Aggregated dataframe
        """
        try:
            result = df.groupby(group_by).agg(agg_dict)
            logger.info(f"Data aggregated by {group_by}")
            return result
        
        except Exception as e:
            logger.error(f"Error aggregating data: {str(e)}")
            return None


class DataValidator:
    """Validate data integrity and format"""
    
    @staticmethod
    def validate_dataframe(df: Any) -> Dict[str, Any]:
        """
        Validate dataframe
        
        Args:
            df: Dataframe to validate
        
        Returns:
            Validation report
        """
        try:
            import pandas as pd
            
            if not isinstance(df, pd.DataFrame):
                return {"valid": False, "error": "Not a DataFrame"}
            
            report = {
                "valid": True,
                "rows": len(df),
                "columns": len(df.columns),
                "missing_values": df.isnull().sum().to_dict(),
                "dtypes": df.dtypes.to_dict(),
                "duplicates": df.duplicated().sum(),
            }
            
            return report
        
        except Exception as e:
            logger.error(f"Error validating dataframe: {str(e)}")
            return {"valid": False, "error": str(e)}
    
    @staticmethod
    def validate_columns(df: Any, required_columns: List[str]) -> bool:
        """
        Check if dataframe has required columns
        
        Args:
            df: Dataframe to check
            required_columns: Required column names
        
        Returns:
            True if all required columns present
        """
        missing = set(required_columns) - set(df.columns)
        if missing:
            logger.warning(f"Missing columns: {missing}")
            return False
        return True


class DataExporter:
    """Export data in various formats"""
    
    @staticmethod
    def export_csv(df: Any, filepath: str) -> bool:
        """
        Export dataframe to CSV
        
        Args:
            df: Dataframe to export
            filepath: Output file path
        
        Returns:
            True if successful
        """
        try:
            df.to_csv(filepath, index=False)
            logger.info(f"Exported to CSV: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error exporting CSV: {str(e)}")
            return False
    
    @staticmethod
    def export_json(data: Dict, filepath: str) -> bool:
        """
        Export data to JSON
        
        Args:
            data: Data to export
            filepath: Output file path
        
        Returns:
            True if successful
        """
        try:
            import json
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Exported to JSON: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error exporting JSON: {str(e)}")
            return False
