"""
Environment Setup Module
Handles directory creation and system path configuration
"""

import os
import sys
from pathlib import Path


class EnvironmentSetup:
    """Handles environment initialization and directory setup"""
    
    def __init__(self):
        self.current_dir = Path.cwd()
        self.data_dir = self.current_dir / "data"
        self.input_dir = self.data_dir / "input"
        self.output_dir = self.data_dir / "output"
    
    def setup_directories(self):
        """Create necessary directories if they don't exist"""
        try:
            directories = [self.data_dir, self.input_dir, self.output_dir]
            created = []
            existing = []
            
            for directory in directories:
                if directory.exists():
                    existing.append(directory.name)
                else:
                    directory.mkdir(parents=True, exist_ok=True)
                    created.append(directory.name)
            
            return {
                'success': True,
                'created': created,
                'existing': existing,
                'paths': {
                    'current': self.current_dir,
                    'input': self.input_dir,
                    'output': self.output_dir
                }
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'paths': None
            }
    
    def setup_python_path(self):
        """Add project directory to Python path"""
        try:
            if str(self.current_dir) not in sys.path:
                sys.path.append(str(self.current_dir))
                return {'success': True, 'added': True}
            else:
                return {'success': True, 'added': False}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_paths(self):
        """Get all configured paths"""
        return {
            'current_dir': self.current_dir,
            'data_dir': self.data_dir,
            'input_dir': self.input_dir,
            'output_dir': self.output_dir
        }
