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
        """Load and parse a financial document"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.csv':
                return self.csv_parser.parse(file_path)
            elif file_extension == '.pdf':
                return self.pdf_parser.parse(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
                
        except Exception as e:
            self.logger.error(f"Error loading file: {str(e)}")
            raise

    def validate_file(self, file_path: str) -> bool:
        """Validate if file exists and is of supported type"""
        if not os.path.exists(file_path):
            return False
            
        supported_extensions = ['.csv', '.pdf']
        file_extension = os.path.splitext(file_path)[1].lower()
        
        return file_extension in supported_extensions
