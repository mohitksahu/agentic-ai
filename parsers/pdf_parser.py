import pandas as pd
import re
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import os

# Try multiple PDF parsing libraries for better compatibility
try:
    import PyPDF2
    from PyPDF2 import PdfReader
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import tabula
    HAS_TABULA = True
except ImportError:
    HAS_TABULA = False

class PDFParser:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Common patterns for financial data
        self.amount_patterns = [
            r'\$\s*\d+[,.]?\d*\.?\d*',           # $123.45, $1,234.56
            r'\d+[,.]\d{2}',                     # 123.45, 123,45
            r'-?\$?\s*\d+[,.]?\d*\.?\d*',        # -$123.45, -123.45
            r'\d{1,3}(?:,\d{3})*(?:\.\d{2})?',   # 1,234.56
        ]
        
        self.date_patterns = [
            r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',    # 12/31/2023, 12-31-23
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',      # 2023-12-31
            r'[A-Za-z]{3}\s+\d{1,2}[,]\s+\d{4}', # Jan 31, 2023
            r'\d{1,2}\s+[A-Za-z]{3}\s+\d{4}',    # 31 Jan 2023
        ]
        
        # Common financial categories/keywords
        self.category_keywords = {
            'Groceries': ['grocery', 'supermarket', 'food', 'walmart', 'target', 'costco', 'safeway'],
            'Dining': ['restaurant', 'cafe', 'bar', 'mcdonald', 'starbucks', 'pizza', 'dining'],
            'Transportation': ['gas', 'fuel', 'uber', 'lyft', 'taxi', 'metro', 'parking', 'toll'],
            'Utilities': ['electric', 'water', 'gas bill', 'internet', 'phone', 'cable', 'utility'],
            'Entertainment': ['movie', 'theater', 'netflix', 'spotify', 'game', 'entertainment'],
            'Healthcare': ['doctor', 'pharmacy', 'hospital', 'medical', 'dental', 'health'],
            'Shopping': ['amazon', 'ebay', 'store', 'shopping', 'mall', 'retail'],
            'Housing': ['rent', 'mortgage', 'apartment', 'housing'],
            'Insurance': ['insurance', 'policy', 'premium'],
            'Banking': ['fee', 'charge', 'interest', 'atm', 'bank']
        }

    def parse(self, file_path: str) -> Dict[str, Any]:
        """Parse a PDF file containing financial data using multiple approaches"""
        try:
            self.logger.info(f"Parsing PDF file: {file_path}")
            
            # Try different parsing methods in order of preference
            extracted_data = None
            parsing_method = None
            
            # Method 1: Try pdfplumber (best for structured data)
            if HAS_PDFPLUMBER and extracted_data is None:
                try:
                    extracted_data = self._parse_with_pdfplumber(file_path)
                    parsing_method = "pdfplumber"
                except Exception as e:
                    self.logger.warning(f"pdfplumber parsing failed: {e}")
            
            # Method 2: Try tabula (good for tables)
            if HAS_TABULA and extracted_data is None:
                try:
                    extracted_data = self._parse_with_tabula(file_path)
                    parsing_method = "tabula"
                except Exception as e:
                    self.logger.warning(f"tabula parsing failed: {e}")
            
            # Method 3: Fallback to PyPDF2 (basic text extraction)
            if HAS_PYPDF2 and extracted_data is None:
                try:
                    extracted_data = self._parse_with_pypdf2(file_path)
                    parsing_method = "PyPDF2"
                except Exception as e:
                    self.logger.warning(f"PyPDF2 parsing failed: {e}")
            
            if extracted_data is None or len(extracted_data) == 0:
                raise ValueError("Could not extract financial data from PDF. Please ensure the PDF contains readable financial transactions.")
            
            # Convert to DataFrame and validate
            df = pd.DataFrame(extracted_data)
            df = self._clean_and_validate_data(df)
            
            if df.empty:
                raise ValueError("No valid financial data found in PDF after processing.")
            
            validation_result = self._validate_financial_data(df)
            
            return {
                'data': df,
                'summary': {
                    'total_rows': len(df),
                    'date_range': [df['date'].min().strftime('%Y-%m-%d'), 
                                  df['date'].max().strftime('%Y-%m-%d')] if not df.empty else [],
                    'total_amount': float(df['amount'].sum()) if not df.empty else 0,
                    'categories': df['category'].unique().tolist() if not df.empty else [],
                    'avg_transaction': float(df['amount'].mean()) if not df.empty else 0,
                    'parsing_method': parsing_method,
                    'data_quality': validation_result['quality_score']
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing PDF file: {str(e)}")
            raise

    def _parse_with_pdfplumber(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse PDF using pdfplumber for better text extraction"""
        extracted_data = []
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    # Try to extract tables first
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            table_data = self._process_table_data(table)
                            extracted_data.extend(table_data)
                    
                    # Extract text and parse for financial data
                    text = page.extract_text()
                    if text:
                        text_data = self._extract_from_text(text)
                        extracted_data.extend(text_data)
                        
                except Exception as e:
                    self.logger.warning(f"Error processing page {page_num}: {e}")
                    continue
        
        return extracted_data

    def _parse_with_tabula(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse PDF using tabula-py for table extraction"""
        extracted_data = []
        
        try:
            # Extract all tables from PDF
            tables = tabula.read_pdf(file_path, pages='all', multiple_tables=True)
            
            for table in tables:
                if not table.empty:
                    table_data = self._process_dataframe_table(table)
                    extracted_data.extend(table_data)
                    
        except Exception as e:
            self.logger.warning(f"Tabula extraction failed: {e}")
        
        return extracted_data

    def _parse_with_pypdf2(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse PDF using PyPDF2 as fallback"""
        extracted_data = []
        
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            
            for page_num, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                    if text:
                        text_data = self._extract_from_text(text)
                        extracted_data.extend(text_data)
                except Exception as e:
                    self.logger.warning(f"Error processing page {page_num}: {e}")
                    continue
        
        return extracted_data

    def _process_table_data(self, table: List[List[str]]) -> List[Dict[str, Any]]:
        """Process extracted table data"""
        processed_data = []
        
        if not table or len(table) < 2:
            return processed_data
        
        # Try to identify header row and data columns
        header = table[0] if table else []
        
        # Look for date, amount, description columns
        date_col_idx = self._find_column_index(header, ['date', 'transaction', 'posted'])
        amount_col_idx = self._find_column_index(header, ['amount', 'debit', 'credit', 'total'])
        desc_col_idx = self._find_column_index(header, ['description', 'merchant', 'details'])
        
        for row in table[1:]:  # Skip header
            if len(row) >= max(filter(None, [date_col_idx, amount_col_idx, desc_col_idx]), default=0):
                try:
                    transaction = {}
                    
                    if date_col_idx is not None and date_col_idx < len(row):
                        transaction['raw_date'] = row[date_col_idx]
                    
                    if amount_col_idx is not None and amount_col_idx < len(row):
                        transaction['raw_amount'] = row[amount_col_idx]
                    
                    if desc_col_idx is not None and desc_col_idx < len(row):
                        transaction['raw_description'] = row[desc_col_idx]
                    else:
                        transaction['raw_description'] = ' '.join(row)
                    
                    if transaction:
                        processed_data.append(transaction)
                        
                except Exception as e:
                    continue
        
        return processed_data

    def _process_dataframe_table(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Process pandas DataFrame from tabula extraction"""
        processed_data = []
        
        # Find relevant columns
        date_col = self._find_dataframe_column(df, ['date', 'transaction', 'posted'])
        amount_col = self._find_dataframe_column(df, ['amount', 'debit', 'credit', 'total'])
        desc_col = self._find_dataframe_column(df, ['description', 'merchant', 'details', 'reference'])
        
        for _, row in df.iterrows():
            try:
                transaction = {}
                
                if date_col:
                    transaction['raw_date'] = str(row[date_col])
                
                if amount_col:
                    transaction['raw_amount'] = str(row[amount_col])
                
                if desc_col:
                    transaction['raw_description'] = str(row[desc_col])
                else:
                    transaction['raw_description'] = ' '.join([str(val) for val in row.values if pd.notna(val)])
                
                if transaction and any(key in transaction for key in ['raw_date', 'raw_amount']):
                    processed_data.append(transaction)
                    
            except Exception as e:
                continue
        
        return processed_data

    def _extract_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract financial data from raw text"""
        extracted_data = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for lines that contain both dates and amounts
            dates = self._extract_dates(line)
            amounts = self._extract_amounts(line)
            
            if dates and amounts:
                # Create transaction for each date-amount pair
                for i in range(min(len(dates), len(amounts))):
                    transaction = {
                        'raw_date': dates[i],
                        'raw_amount': amounts[i],
                        'raw_description': line
                    }
                    extracted_data.append(transaction)
        
        return extracted_data

    def _find_column_index(self, header: List[str], keywords: List[str]) -> Optional[int]:
        """Find column index by matching keywords"""
        for i, col in enumerate(header):
            if any(keyword.lower() in col.lower() for keyword in keywords):
                return i
        return None

    def _find_dataframe_column(self, df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
        """Find DataFrame column by matching keywords"""
        for col in df.columns:
            if any(keyword.lower() in str(col).lower() for keyword in keywords):
                return col
        return None

    def _extract_amounts(self, text: str) -> List[str]:
        """Extract monetary amounts from text using multiple patterns"""
        amounts = []
        for pattern in self.amount_patterns:
            matches = re.findall(pattern, text)
            amounts.extend(matches)
        return amounts

    def _extract_dates(self, text: str) -> List[str]:
        """Extract dates from text using multiple patterns"""
        dates = []
        for pattern in self.date_patterns:
            matches = re.findall(pattern, text)
            dates.extend(matches)
        return dates

    def _categorize_transaction(self, description: str) -> str:
        """Categorize transaction based on description"""
        description_lower = description.lower()
        
        for category, keywords in self.category_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                return category
        
        return 'Uncategorized'

    def _clean_and_validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate extracted data"""
        cleaned_data = []
        
        for _, row in df.iterrows():
            try:
                # Process date
                if 'raw_date' in row and pd.notna(row['raw_date']):
                    try:
                        date = pd.to_datetime(row['raw_date'], errors='coerce')
                        if pd.isna(date):
                            continue
                    except:
                        continue
                else:
                    continue
                
                # Process amount
                if 'raw_amount' in row and pd.notna(row['raw_amount']):
                    try:
                        amount_str = str(row['raw_amount'])
                        # Clean amount string
                        amount_clean = re.sub(r'[^\d.-]', '', amount_str)
                        amount = float(amount_clean)
                        if amount <= 0:
                            continue
                    except:
                        continue
                else:
                    continue
                
                # Process description/category
                description = str(row.get('raw_description', 'Unknown Transaction'))
                category = self._categorize_transaction(description)
                
                cleaned_data.append({
                    'date': date,
                    'amount': abs(amount),  # Use absolute value
                    'category': category,
                    'description': description
                })
                
            except Exception as e:
                continue
        
        return pd.DataFrame(cleaned_data)

    def _validate_financial_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate the quality of extracted financial data"""
        quality_score = 100
        warnings = []
        
        if len(df) < 3:
            warnings.append("Very few transactions extracted")
            quality_score -= 30
        
        if df['amount'].sum() == 0:
            warnings.append("No valid amounts found")
            quality_score -= 50
        
        unique_categories = df['category'].nunique()
        if unique_categories < 2:
            warnings.append("Limited category diversity")
            quality_score -= 15
        
        # Check date range
        if len(df) > 0:
            date_range = (df['date'].max() - df['date'].min()).days
            if date_range < 1:
                warnings.append("All transactions on same date")
                quality_score -= 20
        
        return {
            'quality_score': max(0, quality_score),
            'warnings': warnings
        }
