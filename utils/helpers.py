import os
import logging
from datetime import datetime
from typing import Any, Dict, List
import pandas as pd

class Helpers:
    @staticmethod
    def setup_logging():
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    @staticmethod
    def format_currency(amount: float) -> str:
        """Format amount as currency string"""
        return f"${amount:,.2f}"

    @staticmethod
    def validate_date_format(date_str: str) -> bool:
        """Validate if string is in correct date format"""
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            return False

    @staticmethod
    def create_directory_if_not_exists(directory: str):
        """Create directory if it doesn't exist"""
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def calculate_percentage(part: float, whole: float) -> float:
        """Calculate percentage with error handling"""
        try:
            return (part / whole * 100) if whole != 0 else 0
        except ZeroDivisionError:
            return 0

    @staticmethod
    def merge_dataframes(dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge multiple dataframes with common columns"""
        if not dfs:
            return pd.DataFrame()
            
        merged_df = dfs[0]
        for df in dfs[1:]:
            merged_df = pd.concat([merged_df, df], ignore_index=True)
            
        return merged_df

    @staticmethod
    def validate_transaction_data(data: Dict[str, Any]) -> bool:
        """Validate transaction data structure"""
        required_keys = ['date', 'amount', 'category']
        return all(key in data for key in required_keys)
