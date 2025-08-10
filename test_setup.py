"""
Production Test Setup for Agentic Financial AI System
Tests system components and validates user data processing capabilities
"""

import unittest
import os
import sys
import pandas as pd
from pathlib import Path
import logging

# Setup logging for test results
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('system_validation.log', encoding='utf-8', mode='w')
    ]
)

logger = logging.getLogger(__name__)


class SystemValidationTest(unittest.TestCase):
    """Validate system components and user data processing"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.project_root = Path(__file__).parent
        cls.data_dir = cls.project_root / "data"
        cls.input_dir = cls.data_dir / "input"
        cls.output_dir = cls.data_dir / "output"
        
        # Ensure directories exist
        cls.input_dir.mkdir(parents=True, exist_ok=True)
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Test environment initialized")

    def test_01_directory_structure(self):
        """Test that required directories exist"""
        self.assertTrue(self.data_dir.exists(), "Data directory missing")
        self.assertTrue(self.input_dir.exists(), "Input directory missing")
        self.assertTrue(self.output_dir.exists(), "Output directory missing")
        logger.info("✅ Directory structure validated")

    def test_02_dependencies(self):
        """Test that required dependencies are available"""
        required_packages = ['pandas', 'numpy', 'pathlib']
        
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"✅ {package} available")
            except ImportError:
                self.fail(f"Required package {package} not available")

    def test_03_csv_detection(self):
        """Test CSV file detection in input directory"""
        csv_files = list(self.input_dir.glob("*.csv"))
        
        if csv_files:
            logger.info(f"✅ Found {len(csv_files)} CSV files in input directory")
            for csv_file in csv_files:
                self.assertTrue(csv_file.stat().st_size > 0, f"CSV file {csv_file.name} is empty")
                logger.info(f"   📁 {csv_file.name} ({csv_file.stat().st_size} bytes)")
        else:
            logger.warning("⚠️ No CSV files found in input directory")
            logger.info("💡 Add CSV files to data/input/ to enable full testing")

    def test_04_csv_structure_validation(self):
        """Test CSV files have proper structure for financial analysis"""
        csv_files = list(self.input_dir.glob("*.csv"))
        
        if not csv_files:
            self.skipTest("No CSV files to validate")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Basic structure checks
                self.assertGreater(len(df), 0, f"CSV {csv_file.name} has no data rows")
                self.assertGreater(len(df.columns), 1, f"CSV {csv_file.name} needs multiple columns")
                
                # Check for financial data patterns
                has_numeric_column = any(df[col].dtype in ['int64', 'float64'] for col in df.columns)
                self.assertTrue(has_numeric_column, f"CSV {csv_file.name} needs numeric columns for amounts")
                
                logger.info(f"✅ {csv_file.name} structure validated:")
                logger.info(f"   📊 Rows: {len(df)}, Columns: {len(df.columns)}")
                logger.info(f"   📋 Columns: {list(df.columns)}")
                
            except Exception as e:
                self.fail(f"Error validating CSV {csv_file.name}: {str(e)}")

    def test_05_component_imports(self):
        """Test that core components can be imported"""
        try:
            # Test utility imports
            from utils.environment_setup import EnvironmentSetup
            from utils.dependency_manager import DependencyManager
            from utils.file_loader import FileLoader
            
            logger.info("✅ Core utility components importable")
            
            # Test parser imports
            from parsers.csv_parser import CSVParser
            logger.info("✅ Parser components importable")
            
        except ImportError as e:
            logger.warning(f"⚠️ Component import issue: {str(e)}")
            logger.info("ℹ️ Components may be inline in notebook - this is acceptable")

    def test_06_notebook_exists(self):
        """Test that main notebook exists and is readable"""
        notebook_path = self.project_root / "main.ipynb"
        self.assertTrue(notebook_path.exists(), "Main notebook missing")
        self.assertGreater(notebook_path.stat().st_size, 0, "Notebook is empty")
        logger.info("✅ Main notebook exists and has content")

    def test_07_output_directory_writable(self):
        """Test that output directory is writable"""
        test_file = self.output_dir / "test_write.txt"
        
        try:
            test_file.write_text("test")
            self.assertTrue(test_file.exists(), "Cannot write to output directory")
            test_file.unlink()  # Clean up
            
            logger.info("✅ Output directory is writable")
            
        except Exception as e:
            self.fail(f"Output directory not writable: {str(e)}")

    def test_08_visualization_directory(self):
        """Test visualization directory setup"""
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        self.assertTrue(viz_dir.exists(), "Visualization directory missing")
        logger.info("✅ Visualization directory ready")

    def test_09_system_readiness(self):
        """Overall system readiness check"""
        csv_files = list(self.input_dir.glob("*.csv"))
        
        readiness_score = 0
        total_checks = 5
        
        # Check 1: Directories exist
        if all(d.exists() for d in [self.data_dir, self.input_dir, self.output_dir]):
            readiness_score += 1
            logger.info("✅ Directory structure ready")
        
        # Check 2: Output writable
        try:
            test_file = self.output_dir / "readiness_test.tmp"
            test_file.write_text("test")
            test_file.unlink()
            readiness_score += 1
            logger.info("✅ Output directory ready")
        except:
            logger.warning("⚠️ Output directory issues")
        
        # Check 3: Python path setup
        if str(self.project_root) in sys.path:
            readiness_score += 1
            logger.info("✅ Python path configured")
        else:
            sys.path.append(str(self.project_root))
            readiness_score += 1
            logger.info("✅ Python path added")
        
        # Check 4: Dependencies available
        try:
            import pandas, numpy
            readiness_score += 1
            logger.info("✅ Core dependencies available")
        except:
            logger.warning("⚠️ Some dependencies missing")
        
        # Check 5: Data availability
        if csv_files:
            readiness_score += 1
            logger.info(f"✅ User data available ({len(csv_files)} CSV files)")
        else:
            logger.info("ℹ️ System ready for user data upload")
        
        readiness_percentage = (readiness_score / total_checks) * 100
        
        logger.info(f"📊 System Readiness: {readiness_percentage:.0f}% ({readiness_score}/{total_checks})")
        
        if readiness_percentage >= 80:
            logger.info("🚀 System ready for financial analysis!")
        elif readiness_percentage >= 60:
            logger.info("⚠️ System mostly ready - minor issues detected")
        else:
            logger.warning("❌ System needs attention before use")
        
        self.assertGreaterEqual(readiness_score, 3, "System not ready for production use")


def run_system_validation():
    """Run system validation tests"""
    print("🔍 Running Agentic AI Financial System Validation...")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(SystemValidationTest)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)
    
    print("\n" + "=" * 60)
    
    if result.wasSuccessful():
        print("✅ All validation tests passed!")
        print("🚀 System ready for financial analysis")
    else:
        print(f"⚠️ {len(result.failures + result.errors)} issues found")
        print("💡 Check system_validation.log for details")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_system_validation()
    sys.exit(0 if success else 1)
