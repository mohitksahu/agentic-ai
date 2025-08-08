import os
from typing import Dict, Any
from parsers.csv_parser import CSVParser
from parsers.pdf_parser import PDFParser
import logging

class FileLoader:
    def __init__(self):
        self.csv_parser = CSVParser()
        self.pdf_parser = PDFParser()
        self.logger = logging.getLogger(__name__)

    def load_file(self, file_path: str) -> Dict[str, Any]:
        """Load and parse a financial document with RAG-enhanced validation"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.csv':
                result = self.csv_parser.parse(file_path)
                self._validate_financial_data(result, file_path)
                return result
            elif file_extension == '.pdf':
                result = self.pdf_parser.parse(file_path)
                self._validate_financial_data(result, file_path)
                return result
            else:
                raise ValueError(f"Unsupported file type: {file_extension}. Please upload CSV or PDF files containing your financial data.")
                
        except Exception as e:
            self.logger.error(f"Error loading file: {str(e)}")
            raise

    def _validate_financial_data(self, result: Dict[str, Any], file_path: str) -> None:
        """Validate that the loaded data is suitable for financial analysis"""
        try:
            df = result['data']
            
            # Check for required columns
            required_info = ['date', 'amount']
            available_cols = df.columns.tolist()
            
            missing_info = []
            for info in required_info:
                if not any(col.lower() in info for col in available_cols):
                    missing_info.append(info)
            
            if missing_info:
                self.logger.warning(f"File {file_path} may be missing: {', '.join(missing_info)}")
            
            # Check data quality
            if len(df) == 0:
                raise ValueError("The uploaded file contains no data. Please ensure your file has financial transaction data.")
            
            if len(df) < 5:
                self.logger.warning(f"File {file_path} has very few transactions ({len(df)}). Analysis may be limited.")
            
            # Log successful validation
            self.logger.info(f"Successfully validated {len(df)} transactions from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error validating financial data: {str(e)}")
            raise ValueError(f"The uploaded file doesn't appear to contain valid financial data: {str(e)}")

    def validate_file(self, file_path: str) -> bool:
        """Validate if file exists and is of supported type for financial analysis"""
        if not os.path.exists(file_path):
            return False
            
        supported_extensions = ['.csv', '.pdf']
        file_extension = os.path.splitext(file_path)[1].lower()
        
        return file_extension in supported_extensions

    def get_supported_formats(self) -> Dict[str, str]:
        """Return information about supported file formats for RAG system"""
        return {
            '.csv': 'CSV files containing financial transaction data (date, amount, category/description columns)',
            '.pdf': 'PDF bank statements and financial reports (automatically parsed for transaction data)'
        }
