import pandas as pd
from typing import Dict, Any, List, Optional
import logging
import numpy as np
from datetime import datetime
import re

class CSVParser:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Common column name variations for different data sources
        self.date_columns = ['date', 'Date', 'DATE', 'transaction_date', 'Transaction Date', 
                            'posted_date', 'Posted Date', 'timestamp', 'Timestamp']
        self.amount_columns = ['amount', 'Amount', 'AMOUNT', 'transaction_amount', 
                              'Transaction Amount', 'debit', 'Debit', 'credit', 'Credit',
                              'value', 'Value', 'sum', 'Sum', 'price', 'Price']
        self.category_columns = ['category', 'Category', 'CATEGORY', 'description', 
                               'Description', 'merchant', 'Merchant', 'type', 'Type',
                               'expense_type', 'Expense Type', 'account', 'Account']

    def parse(self, file_path: str) -> Dict[str, Any]:
        """Parse a CSV file containing financial data with intelligent column detection"""
        try:
            # Try different encodings and separators
            df = self._read_csv_robust(file_path)
            
            if df is None or df.empty:
                raise ValueError("Could not read CSV file or file is empty")
            
            self.logger.info(f"Successfully read CSV with {len(df)} rows and {len(df.columns)} columns")
            
            # Detect and map columns
            column_mapping = self._detect_columns(df)
            
            if not column_mapping:
                raise ValueError("Could not identify financial data columns. Please ensure your CSV has date, amount, and category/description columns.")
            
            # Map columns to standard names
            df_processed = self._process_columns(df, column_mapping)
            
            # Validate the processed data
            validation_result = self._validate_financial_data(df_processed)
            
            if not validation_result['is_valid']:
                raise ValueError(f"Invalid financial data: {validation_result['errors']}")
            
            return {
                'data': df_processed,
                'summary': {
                    'total_rows': len(df_processed),
                    'date_range': [df_processed['date'].min().strftime('%Y-%m-%d'), 
                                  df_processed['date'].max().strftime('%Y-%m-%d')],
                    'total_amount': float(df_processed['amount'].sum()),
                    'categories': df_processed['category'].unique().tolist(),
                    'avg_transaction': float(df_processed['amount'].mean()),
                    'column_mapping': column_mapping,
                    'data_quality': validation_result['quality_score']
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing CSV file: {str(e)}")
            raise

    def _read_csv_robust(self, file_path: str) -> Optional[pd.DataFrame]:
        """Try multiple encoding and separator combinations to read CSV"""
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        separators = [',', ';', '\t', '|']
        
        for encoding in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, separator=sep, 
                                   skip_blank_lines=True, na_values=['', 'N/A', 'NULL', 'nan'])
                    
                    # Check if we got meaningful data
                    if len(df.columns) > 1 and len(df) > 0:
                        self.logger.info(f"Successfully read with encoding={encoding}, separator='{sep}'")
                        return df
                        
                except Exception as e:
                    continue
        
        # Try Excel format as backup
        try:
            df = pd.read_excel(file_path)
            self.logger.info("File read as Excel format")
            return df
        except:
            pass
            
        return None

    def _detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Intelligently detect which columns contain date, amount, and category data"""
        column_mapping = {}
        
        # Detect date column
        date_col = self._find_column_by_content(df, self.date_columns, 'date')
        if date_col:
            column_mapping['date'] = date_col
        
        # Detect amount column
        amount_col = self._find_column_by_content(df, self.amount_columns, 'numeric')
        if amount_col:
            column_mapping['amount'] = amount_col
        
        # Detect category/description column
        category_col = self._find_column_by_content(df, self.category_columns, 'text')
        if category_col:
            column_mapping['category'] = category_col
        
        return column_mapping

    def _find_column_by_content(self, df: pd.DataFrame, possible_names: List[str], 
                               expected_type: str) -> Optional[str]:
        """Find column by matching names and validating content type"""
        
        # First try exact name matches
        for col_name in possible_names:
            if col_name in df.columns:
                if self._validate_column_type(df[col_name], expected_type):
                    return col_name
        
        # Then try partial matches and content validation
        for col in df.columns:
            if any(name.lower() in col.lower() for name in possible_names):
                if self._validate_column_type(df[col], expected_type):
                    return col
        
        # Finally, check all columns for the expected data type
        for col in df.columns:
            if self._validate_column_type(df[col], expected_type):
                return col
        
        return None

    def _validate_column_type(self, series: pd.Series, expected_type: str) -> bool:
        """Validate if a column contains the expected data type"""
        # Remove null values for validation
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return False
        
        if expected_type == 'date':
            return self._is_date_column(clean_series)
        elif expected_type == 'numeric':
            return self._is_numeric_column(clean_series)
        elif expected_type == 'text':
            return self._is_text_column(clean_series)
        
        return False

    def _is_date_column(self, series: pd.Series) -> bool:
        """Check if series contains date-like data"""
        sample_size = min(10, len(series))
        sample = series.head(sample_size)
        
        date_count = 0
        for value in sample:
            try:
                pd.to_datetime(str(value))
                date_count += 1
            except:
                pass
        
        return date_count / sample_size > 0.7

    def _is_numeric_column(self, series: pd.Series) -> bool:
        """Check if series contains numeric/monetary data"""
        sample_size = min(10, len(series))
        sample = series.head(sample_size)
        
        numeric_count = 0
        for value in sample:
            try:
                # Remove currency symbols and convert to float
                clean_val = re.sub(r'[\$£€¥,\s]', '', str(value))
                float(clean_val)
                numeric_count += 1
            except:
                pass
        
        return numeric_count / sample_size > 0.7

    def _is_text_column(self, series: pd.Series) -> bool:
        """Check if series contains text data suitable for categories"""
        sample_size = min(10, len(series))
        sample = series.head(sample_size)
        
        text_count = 0
        for value in sample:
            if isinstance(value, str) and len(str(value).strip()) > 0:
                text_count += 1
        
        return text_count / sample_size > 0.5

    def _process_columns(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
        """Process and standardize the detected columns"""
        processed_df = pd.DataFrame()
        
        # Process date column
        if 'date' in column_mapping:
            date_col = column_mapping['date']
            processed_df['date'] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Process amount column
        if 'amount' in column_mapping:
            amount_col = column_mapping['amount']
            # Clean and convert amount data
            processed_df['amount'] = df[amount_col].astype(str).str.replace(r'[\$£€¥,\s()]', '', regex=True)
            processed_df['amount'] = pd.to_numeric(processed_df['amount'], errors='coerce').abs()
        
        # Process category column
        if 'category' in column_mapping:
            category_col = column_mapping['category']
            processed_df['category'] = df[category_col].astype(str).str.strip()
            # Standardize common categories
            processed_df['category'] = processed_df['category'].apply(self._standardize_category)
        else:
            processed_df['category'] = 'Uncategorized'
        
        # Remove rows with invalid data
        processed_df = processed_df.dropna(subset=['date', 'amount'])
        
        return processed_df

    def _standardize_category(self, category: str) -> str:
        """Standardize category names"""
        category = category.lower().strip()
        
        # Common category mappings
        category_map = {
            'grocery': 'Groceries', 'food': 'Groceries', 'supermarket': 'Groceries',
            'restaurant': 'Dining', 'dining': 'Dining', 'cafe': 'Dining',
            'gas': 'Transportation', 'fuel': 'Transportation', 'uber': 'Transportation',
            'electric': 'Utilities', 'water': 'Utilities', 'internet': 'Utilities',
            'movie': 'Entertainment', 'netflix': 'Entertainment', 'spotify': 'Entertainment',
            'doctor': 'Healthcare', 'pharmacy': 'Healthcare', 'hospital': 'Healthcare',
            'amazon': 'Shopping', 'store': 'Shopping', 'mall': 'Shopping',
            'rent': 'Housing', 'mortgage': 'Housing', 'insurance': 'Insurance'
        }
        
        for key, standard_name in category_map.items():
            if key in category:
                return standard_name
        
        return category.title()

    def _validate_financial_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate the quality and completeness of financial data"""
        errors = []
        quality_score = 100
        
        # Check for required columns
        required_cols = ['date', 'amount', 'category']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            quality_score -= 50
        
        # Check data completeness
        if len(df) < 5:
            errors.append("Dataset too small (less than 5 transactions)")
            quality_score -= 30
        
        # Check date range
        if 'date' in df.columns:
            date_range = (df['date'].max() - df['date'].min()).days
            if date_range < 7:
                errors.append("Date range too short (less than 1 week)")
                quality_score -= 20
        
        # Check amount validity
        if 'amount' in df.columns:
            zero_amounts = (df['amount'] == 0).sum()
            if zero_amounts > len(df) * 0.5:
                errors.append("Too many zero amounts")
                quality_score -= 15
            
            if df['amount'].max() > 50000:
                errors.append("Warning: Very large transaction amounts detected")
                quality_score -= 5
        
        # Check category diversity
        if 'category' in df.columns:
            unique_categories = df['category'].nunique()
            if unique_categories < 2:
                errors.append("Warning: Limited category diversity")
                quality_score -= 10
        
        is_valid = len([e for e in errors if not e.startswith("Warning")]) == 0
        
        return {
            'is_valid': is_valid,
            'errors': errors,
            'quality_score': max(0, quality_score)
        }
