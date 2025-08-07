import os
import shutil
import logging
from pathlib import Path
import subprocess
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_installed_packages():
    """Get list of packages installed in the current environment"""
    try:
        import pkg_resources
        return {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    except Exception as e:
        logger.error(f"Error getting installed packages: {str(e)}")
        return {}

def uninstall_project_packages():
    """Uninstall packages that were installed for this project"""
    project_packages = [
        'transformers',
        'torch',
        'accelerate',
        'huggingface-hub',
        'langchain',
        'pandas',
        'numpy',
        'plotly',
        'pypdf2',
        'python-dotenv',
        'streamlit',
        'jupyter',
        'sentence-transformers',
        'chromadb',
        'pydantic',
        'psutil'
    ]
    
    logger.info("Uninstalling project packages...")
    for package in project_packages:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', package],
                         check=True,
                         capture_output=True)
            logger.info(f"Uninstalled {package}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to uninstall {package}: {e.stderr.decode()}")

def remove_project_directories():
    """Remove project-specific directories"""
    project_root = str(Path().absolute())
    
    # Directories to remove
    dirs_to_remove = [
        os.path.join(project_root, 'models'),  # Downloaded model files
        os.path.join(project_root, '.pytest_cache'),  # Pytest cache
        os.path.join(project_root, '__pycache__'),  # Python cache
        os.path.join(project_root, '.ipynb_checkpoints'),  # Jupyter checkpoints
        os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')  # HuggingFace cache
    ]
    
    for dir_path in dirs_to_remove:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                logger.info(f"Removed directory: {dir_path}")
            except Exception as e:
                logger.error(f"Error removing directory {dir_path}: {str(e)}")

def remove_generated_files():
    """Remove generated data files but keep original source code"""
    project_root = str(Path().absolute())
    
    # Files to remove
    files_to_remove = [
        os.path.join(project_root, 'data', 'test_budget.csv'),
        os.path.join(project_root, 'visualizations', 'output', '*.html'),  # Generated charts
    ]
    
    for file_pattern in files_to_remove:
        try:
            for file_path in Path().glob(file_pattern):
                os.remove(file_path)
                logger.info(f"Removed file: {file_path}")
        except Exception as e:
            logger.error(f"Error removing file {file_pattern}: {str(e)}")

def clear_jupyter_kernel():
    """Remove Jupyter kernel and notebook outputs"""
    try:
        # Remove notebook outputs
        notebook_path = os.path.join(str(Path().absolute()), 'main.ipynb')
        if os.path.exists(notebook_path):
            import nbformat
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            # Clear outputs from all cells
            for cell in nb.cells:
                if 'outputs' in cell:
                    cell.outputs = []
                if 'execution_count' in cell:
                    cell.execution_count = None
            
            # Save cleaned notebook
            with open(notebook_path, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)
            logger.info("Cleared Jupyter notebook outputs")
    
    except Exception as e:
        logger.error(f"Error clearing Jupyter notebook: {str(e)}")

def main():
    """Main cleanup function"""
    logger.info("Starting cleanup process...")
    
    # Get user confirmation
    confirm = input("This will remove all installed packages and downloaded models. Continue? (y/n): ")
    if confirm.lower() != 'y':
        logger.info("Cleanup cancelled.")
        return
    
    try:
        # Perform cleanup steps
        uninstall_project_packages()
        remove_project_directories()
        remove_generated_files()
        clear_jupyter_kernel()
        
        logger.info("\nCleanup completed successfully!")
        logger.info("To start fresh, you can now delete the project directory if needed.")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        logger.error("Some components may not have been properly removed.")

if __name__ == "__main__":
    main()
