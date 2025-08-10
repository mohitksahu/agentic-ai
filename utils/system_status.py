"""
System Status Monitor Module
Provides comprehensive system status checking and reporting
"""

from pathlib import Path
import pandas as pd


class SystemStatusMonitor:
    """Monitors and reports on system component status"""
    
    def __init__(self):
        self.status_data = {}
    
    def check_data_availability(self, combined_financial_data=None):
        """Check if financial data is available and valid"""
        try:
            if combined_financial_data is None:
                return {
                    'available': False,
                    'reason': 'No data provided',
                    'record_count': 0,
                    'columns': []
                }
            
            if not isinstance(combined_financial_data, pd.DataFrame):
                return {
                    'available': False,
                    'reason': f'Invalid data type: {type(combined_financial_data)}',
                    'record_count': 0,
                    'columns': []
                }
            
            if combined_financial_data.empty:
                return {
                    'available': False,
                    'reason': 'Data is empty',
                    'record_count': 0,
                    'columns': list(combined_financial_data.columns)
                }
            
            return {
                'available': True,
                'reason': 'Valid data available',
                'record_count': len(combined_financial_data),
                'columns': list(combined_financial_data.columns),
                'data_shape': combined_financial_data.shape
            }
            
        except Exception as e:
            return {
                'available': False,
                'reason': f'Error checking data: {str(e)}',
                'record_count': 0,
                'columns': []
            }
    
    def check_component_status(self, required_components):
        """Check status of required system components"""
        status = {}
        
        for component_name, component_var in required_components.items():
            try:
                if component_var is not None:
                    status[component_name] = {
                        'available': True,
                        'type': type(component_var).__name__,
                        'status': 'Ready'
                    }
                else:
                    status[component_name] = {
                        'available': False,
                        'type': 'None',
                        'status': 'Not initialized'
                    }
            except Exception as e:
                status[component_name] = {
                    'available': False,
                    'type': 'Error',
                    'status': f'Error: {str(e)}'
                }
        
        return status
    
    def check_file_outputs(self, output_dir):
        """Check what output files have been generated"""
        try:
            output_path = Path(output_dir)
            if not output_path.exists():
                return {'exists': False, 'files': []}
            
            files = {
                'csv_files': list(output_path.glob('*.csv')),
                'html_files': list(output_path.glob('*.html')),
                'txt_files': list(output_path.glob('*.txt')),
                'json_files': list(output_path.glob('*.json')),
                'all_files': list(output_path.iterdir())
            }
            
            return {
                'exists': True,
                'file_counts': {k: len(v) for k, v in files.items()},
                'files': {k: [f.name for f in v] for k, v in files.items()},
                'total_files': len(files['all_files'])
            }
            
        except Exception as e:
            return {
                'exists': False, 
                'error': str(e),
                'files': []
            }
    
    def generate_system_report(self, **kwargs):
        """Generate comprehensive system status report"""
        report = {
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'system_ready': True,
            'issues': [],
            'components': {}
        }
        
        # Check data availability
        if 'combined_financial_data' in kwargs:
            data_status = self.check_data_availability(kwargs['combined_financial_data'])
            report['data_status'] = data_status
            if not data_status['available']:
                report['system_ready'] = False
                report['issues'].append(f"Data issue: {data_status['reason']}")
        
        # Check components
        if 'components' in kwargs:
            component_status = self.check_component_status(kwargs['components'])
            report['component_status'] = component_status
            
            for comp_name, comp_info in component_status.items():
                if not comp_info['available']:
                    report['issues'].append(f"Component issue: {comp_name} - {comp_info['status']}")
        
        # Check outputs
        if 'output_dir' in kwargs:
            output_status = self.check_file_outputs(kwargs['output_dir'])
            report['output_status'] = output_status
        
        return report
    
    def print_system_summary(self, report):
        """Print a formatted system status summary"""
        print("🎉 Financial Analysis System Status")
        print("=" * 50)
        
        # Overall status
        status_icon = "✅" if report['system_ready'] else "⚠️"
        print(f"{status_icon} System Status: {'Ready' if report['system_ready'] else 'Issues Found'}")
        
        # Data status
        if 'data_status' in report:
            data = report['data_status']
            data_icon = "✅" if data['available'] else "❌"
            print(f"{data_icon} Data: {data['record_count']} records, {len(data['columns'])} columns")
        
        # Component status
        if 'component_status' in report:
            components = report['component_status']
            ready_count = sum(1 for comp in components.values() if comp['available'])
            total_count = len(components)
            print(f"🔧 Components: {ready_count}/{total_count} ready")
        
        # Output files
        if 'output_status' in report:
            outputs = report['output_status']
            if outputs['exists']:
                total_files = outputs['total_files']
                print(f"📁 Output Files: {total_files} files generated")
        
        # Issues
        if report['issues']:
            print("\n⚠️ Issues Found:")
            for issue in report['issues']:
                print(f"   • {issue}")
        
        print(f"\n📊 Report generated at: {report['timestamp']}")
    
    def get_troubleshooting_info(self):
        """Get troubleshooting information for common issues"""
        return {
            'data_issues': [
                'Ensure CSV files are in the data/input/ directory',
                'Check that CSV files have date, amount, and category columns',
                'Verify CSV format is valid and readable'
            ],
            'component_issues': [
                'Run environment setup cell first',
                'Install dependencies if missing',
                'Check for module import errors'
            ],
            'general_tips': [
                'Run cells in order from top to bottom',
                'Wait for each cell to complete before running the next',
                'Check console output for error messages'
            ]
        }
