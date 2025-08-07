import os
import sys
from pathlib import Path
import unittest
import pandas as pd
from huggingface_hub import hf_hub_download
import logging
import torch
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to system path
project_root = str(Path().absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

from llms.gpt2_wrapper import GPT2Wrapper
from llms.distilbert_wrapper import DistilBertWrapper
from agents.financial_agent_simplified import FinancialAgent
from utils.file_loader import FileLoader
from utils.helpers import Helpers

class FinancialBotSetupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.models_dir = os.path.join(project_root, 'models')
        cls.data_dir = os.path.join(project_root, 'data')
        cls.test_csv_path = os.path.join(cls.data_dir, 'test_budget.csv')
        
        # Create necessary directories
        os.makedirs(cls.models_dir, exist_ok=True)
        os.makedirs(cls.data_dir, exist_ok=True)
        
        # Create test CSV if it doesn't exist
        if not os.path.exists(cls.test_csv_path):
            cls._create_test_csv()

    @staticmethod
    def _create_test_csv():
        """Create a test CSV file with sample data"""
        data = {
            'date': pd.date_range(start='2025-01-01', periods=10),
            'amount': [100, 50, 75, 200, 25, 150, 80, 90, 120, 60],
            'category': ['Groceries', 'Transport', 'Utilities', 'Rent', 
                        'Entertainment', 'Groceries', 'Healthcare', 
                        'Transport', 'Utilities', 'Entertainment']
        }
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(project_root, 'data', 'test_budget.csv'), index=False)

    def test_01_system_requirements(self):
        """Test system requirements"""
        logger.info("Testing system requirements...")
        
        # Check Python version
        self.assertGreaterEqual(sys.version_info[:2], (3, 8), 
                              "Python 3.8 or higher is required")
        
        # Check CUDA availability (warning only)
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available. Using CPU (this will be slower)")
        
        # Check available RAM
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        self.assertGreaterEqual(ram_gb, 4, 
                              f"4GB RAM required, only {ram_gb:.1f}GB available")
        
        # Check disk space
        disk = shutil.disk_usage(project_root)
        free_space_gb = disk.free / (1024**3)
        self.assertGreaterEqual(free_space_gb, 5, 
                              f"5GB free space required, only {free_space_gb:.1f}GB available")

    def test_02_download_models(self):
        """Test model downloading"""
        logger.info("Downloading and setting up models...")
        
        models = {
            'gpt2': "gpt2",
            'distilbert': "distilbert-base-uncased"
        }
        
        for model_name, model_id in models.items():
            try:
                # Download model files
                logger.info(f"Downloading {model_name} model...")
                cache_dir = os.path.join(self.models_dir, model_name)
                hf_hub_download(
                    repo_id=model_id,
                    filename="config.json",
                    cache_dir=cache_dir
                )
                self.assertTrue(os.path.exists(cache_dir), 
                              f"Failed to download {model_name} model")
            except Exception as e:
                self.fail(f"Error downloading {model_name} model: {str(e)}")

    def test_03_initialize_models(self):
        """Test model initialization"""
        logger.info("Testing model initialization...")
        
        try:
            # Initialize GPT2 wrapper
            gpt2_model = GPT2Wrapper("gpt2")
            self.assertIsNotNone(gpt2_model, "Failed to initialize GPT2 model")
            
            # Initialize DistilBERT wrapper
            distilbert_model = DistilBertWrapper("distilbert-base-uncased")
            self.assertIsNotNone(distilbert_model, "Failed to initialize DistilBERT model")
            
            # Create agent with GPT2
            agent = FinancialAgent(gpt2_model)
            self.assertIsNotNone(agent, "Failed to create Financial Agent")
            
        except Exception as e:
            self.fail(f"Error initializing models: {str(e)}")

    def test_04_file_processing(self):
        """Test file processing capabilities"""
        logger.info("Testing file processing...")
        
        try:
            # Initialize components
            file_loader = FileLoader()
            
            # Test CSV processing
            data = file_loader.load_file(self.test_csv_path)
            self.assertIsNotNone(data, "Failed to load test CSV file")
            self.assertIn('data', data, "Missing 'data' in processed file")
            self.assertIn('summary', data, "Missing 'summary' in processed file")
            
        except Exception as e:
            self.fail(f"Error in file processing: {str(e)}")

    def test_05_basic_analysis(self):
        """Test basic financial analysis"""
        logger.info("Testing basic financial analysis...")
        
        try:
            # Initialize agent with GPT2
            gpt2_model = GPT2Wrapper("gpt2")
            agent = FinancialAgent(gpt2_model)
            
            # Load test data
            agent.run(f"Process the document {self.test_csv_path}")
            
            # Set test income
            response = agent.run("Set monthly income to 5000")
            self.assertIsNotNone(response, "Failed to set monthly income")
            
            # Test budget analysis
            response = agent.run("Analyze my budget")
            self.assertIsNotNone(response, "Failed to perform budget analysis")
            
            # Try with DistilBERT as well
            logger.info("Testing with DistilBERT model...")
            distilbert_model = DistilBertWrapper("distilbert-base-uncased")
            agent = FinancialAgent(distilbert_model)
            response = agent.run("Analyze my budget")
            self.assertIsNotNone(response, "Failed to perform budget analysis with DistilBERT")
            
        except Exception as e:
            self.fail(f"Error in basic analysis: {str(e)}")

def run_tests():
    """Run all tests and return True if all passed"""
    logger.info("Starting Financial Bot Setup Tests...")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(FinancialBotSetupTest)
    
    # Run tests
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    if result.wasSuccessful():
        logger.info("\nAll tests passed! The Financial Bot is ready to use.")
        logger.info("\nYou can now run the main notebook (main.ipynb)")
        return True
    else:
        logger.error("\nSome tests failed. Please fix the issues before proceeding.")
        return False

if __name__ == "__main__":
    run_tests()
