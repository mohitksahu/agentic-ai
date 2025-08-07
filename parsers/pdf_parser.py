from PyPDF2 import PdfReader
import re
from typing import Dict, Any
import logging
import pandas as pd

class PDFParser:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def parse(self, file_path: str) -> Dict[str, Any]:
        """Parse a PDF file containing financial data (e.g., receipts, statements)"""
        try:
            reader = PdfReader(file_path)
            extracted_data = []
            
            for page in reader.pages:
                text = page.extract_text()
                
                # Extract financial data using regex patterns
                amounts = self._extract_amounts(text)
                dates = self._extract_dates(text)
                categories = self._extract_categories(text)
                
                # Combine extracted data
                for i in range(max(len(amounts), len(dates))):
                    extracted_data.append({
                        'date': dates[i] if i < len(dates) else None,
                        'amount': amounts[i] if i < len(amounts) else None,
                        'category': categories[i] if i < len(categories) else 'Uncategorized'
                    })
            
            # Convert to DataFrame for consistency with CSV parser
            df = pd.DataFrame(extracted_data)
            
            return {
                'data': df,
                'summary': {
                    'total_rows': len(df),
                    'total_amount': df['amount'].sum() if not df.empty else 0,
                    'categories': df['category'].unique().tolist() if not df.empty else []
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing PDF file: {str(e)}")
            raise

    def _extract_amounts(self, text: str) -> list:
        """Extract monetary amounts from text"""
        amount_pattern = r'\$?\d+,?\d*\.?\d*'
        return [float(amount.replace('$', '').replace(',', '')) 
                for amount in re.findall(amount_pattern, text)]

    def _extract_dates(self, text: str) -> list:
        """Extract dates from text"""
        date_pattern = r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}'
        return re.findall(date_pattern, text)

    def _extract_categories(self, text: str) -> list:
        """Extract spending categories from text"""
        # Define common spending categories
        categories = ['Groceries', 'Utilities', 'Entertainment', 'Transportation',
                     'Healthcare', 'Shopping', 'Restaurants', 'Other']
        
        found_categories = []
        for category in categories:
            if category.lower() in text.lower():
                found_categories.append(category)
        
        return found_categories if found_categories else ['Uncategorized']
