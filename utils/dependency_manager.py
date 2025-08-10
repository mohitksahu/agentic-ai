"""
Dependency Manager Module
Handles package installation and dependency checks
"""

import subprocess
import sys
import importlib


class DependencyManager:
    """Manages package installation and dependency checking"""
    
    def __init__(self):
        self.required_packages = [
            'pandas', 'numpy', 'matplotlib', 'seaborn', 
            'plotly', 'transformers', 'torch'
        ]
    
    def install_packages(self, packages=None, quiet=True):
        """Install required packages using pip"""
        try:
            if packages is None:
                packages = self.required_packages
            
            # Prepare pip command
            cmd = [sys.executable, '-m', 'pip', 'install'] + packages
            if quiet:
                cmd.append('-q')
            
            # Run installation
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            return {
                'success': result.returncode == 0,
                'packages': packages,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'packages': packages
            }
    
    def check_dependencies(self, packages=None):
        """Check if required packages are available"""
        if packages is None:
            packages = self.required_packages
        
        results = {}
        for package in packages:
            try:
                importlib.import_module(package)
                results[package] = {'available': True, 'error': None}
            except ImportError as e:
                results[package] = {'available': False, 'error': str(e)}
        
        return results
    
    def get_missing_packages(self, packages=None):
        """Get list of missing packages"""
        check_results = self.check_dependencies(packages)
        return [pkg for pkg, info in check_results.items() if not info['available']]
