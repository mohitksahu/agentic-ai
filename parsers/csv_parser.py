import pandas as pd
from typing import Dict, Any
import logging

class CSVParser:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def parse(self, file_path: str) -> Dict[str, Any]:
        """Parse a CSV file containing financial data"""
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Basic validation
            required_columns = ['date', 'amount', 'category']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Convert amount to numeric, handling currency symbols
            df['amount'] = df['amount'].replace('[\$,]', '', regex=True).astype(float)
            
            return {
                'data': df,
                'summary': {
                    'total_rows': len(df),
                    'date_range': [df['date'].min(), df['date'].max()],
                    'total_amount': df['amount'].sum(),
                    'categories': df['category'].unique().tolist()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing CSV file: {str(e)}")
            raise
