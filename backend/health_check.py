"""
System health checking for the RAG system.
Validates configuration, dependencies, and system components.
"""

import logging
from typing import Dict, Any
from pathlib import Path
import sys

# Try to import config, handle gracefully if not available
try:
    from backend.config import Config
except ImportError:
    Config = None
    print("Warning: Config not available for health checking")

logger = logging.getLogger(__name__)

class HealthChecker:
    """Comprehensive system health monitoring."""

    def __init__(self, config=None):
        """Initialize health checker with optional config."""
        self.config = config or (Config() if Config else None)

    def check_configuration(self) -> Dict[str, Any]:
        """Validate configuration system."""
        try:
            if not self.config:
                return {
                    'status': 'error',
                    'message': 'Configuration system not available',
                    'details': {'config_class_available': False}
                }

            # Check if config loaded successfully
            errors = self.config.validate()

            if not errors:
                return {
                    'status': 'healthy',
                    'message': 'Configuration valid and loaded',
                    'details': {
                        'config_file': self.config.config_path,
                        'development_mode': self.config.development,
                        'documents_folder': self.config.documents_folder
                    }
                }
            else:
                # Determine severity based on development mode
                if self.config.development:
                    return {
                        'status': 'warning',
                        'message': f'Configuration warnings in development mode',
                        'details': {
                            'warnings': errors,
                            'development_mode': True
                        }
                    }
                else:
                    return {
                        'status': 'error',
                        'message': f'Configuration errors in production mode',
                        'details': {
                            'errors': errors,
                            'development_mode': False
                        }
                    }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Configuration check failed: {str(e)}',
                'details': {'exception': str(e)}
            }

    def check_documents_folder(self) -> Dict[str, Any]:
        """Check documents folder accessibility and content."""
        try:
            # Get documents folder path
            if self.config:
                docs_path = Path(self.config.documents_folder)
            else:
                docs_path = Path("documents")

            # Check if folder exists
            if not docs_path.exists():
                return {
                    'status': 'warning',
                    'message': f'Documents folder not found: {docs_path}',
                    'details': {
                        'path': str(docs_path),
                        'exists': False,
                        'suggestion': f'Create with: mkdir -p {docs_path}'
                    }
                }

            # Check if folder is readable
            if not docs_path.is_dir():
                return {
                    'status': 'error',
                    'message': f'Documents path is not a directory: {docs_path}',
                    'details': {'path': str(docs_path), 'is_directory': False}
                }

            # Count supported document types
            supported_extensions = ['.txt', '.md', '.pdf', '.docx']
            doc_files = []
            extension_counts = {}

            for ext in supported_extensions:
                files = list(docs_path.glob(f'*{ext}'))
                doc_files.extend(files)
                extension_counts[ext] = len(files)

            return {
                'status': 'healthy',
                'message': f'Documents folder accessible with {len(doc_files)} supported files',
                'details': {
                    'path': str(docs_path),
                    'exists': True,
                    'total_documents': len(doc_files),
                    'by_extension': extension_counts,
                    'supported_types': supported_extensions
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Documents folder check failed: {str(e)}',
                'details': {'exception': str(e)}
            }

    def check_dependencies(self) -> Dict[str, Any]:
        """Check availability of Python dependencies."""
        try:
            # Define dependency categories
            core_dependencies = ['pathlib', 'logging', 'typing']
            yaml_dependencies = ['yaml']
            rag_dependencies = ['sentence_transformers', 'transformers', 'torch']
            document_dependencies = ['PyPDF2', 'docx']
            vector_dependencies = ['weaviate', 'haystack']

            all_dependencies = {
                'core': core_dependencies,
                'yaml': yaml_dependencies,
                'rag': rag_dependencies,
                'documents': document_dependencies,
                'vector': vector_dependencies
            }

            # Check each category
            results = {}
            missing_critical = []
            missing_optional = []

            for category, deps in all_dependencies.items():
                available = []
                missing = []

                for dep in deps:
                    try:
                        __import__(dep)
                        available.append(dep)
                    except ImportError:
                        missing.append(dep)

                        # Categorize missing dependencies
                        if category in ['core', 'yaml']:
                            missing_critical.append(dep)
                        else:
                            missing_optional.append(dep)

                results[category] = {
                    'available': available,
                    'missing': missing,
                    'status': 'complete' if not missing else 'partial'
                }

            # Determine overall status
            if missing_critical:
                status = 'error'
                message = f'Critical dependencies missing: {missing_critical}'
            elif missing_optional:
                status = 'warning'
                message = f'Optional dependencies missing: {missing_optional}'
            else:
                status = 'healthy'
                message = 'All dependencies available'

            return {
                'status': status,
                'message': message,
                'details': {
                    'by_category': results,
                    'missing_critical': missing_critical,
                    'missing_optional': missing_optional,
                    'python_version': sys.version.split()[0]
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Dependency check failed: {str(e)}',
                'details': {'exception': str(e)}
            }

    def check_legacy_scripts(self) -> Dict[str, Any]:
        """Check availability of existing legacy scripts."""
        try:
            legacy_scripts = [
                'weaviate_rag_pipeline_transformers.py',
                'domain_loader.py',
                'debug_chunks.py'
            ]

            found_scripts = []
            missing_scripts = []
            script_info = {}

            for script in legacy_scripts:
                script_path = Path(script)
                if script_path.exists():
                    found_scripts.append(script)
                    # Get basic file info
                    try:
                        stat = script_path.stat()
                        script_info[script] = {
                            'size_bytes': stat.st_size,
                            'size_kb': round(stat.st_size / 1024, 1)
                        }
                    except:
                        script_info[script] = {'size': 'unknown'}
                else:
                    missing_scripts.append(script)

            # Determine status
            if len(found_scripts) == len(legacy_scripts):
                status = 'healthy'
                message = f'All {len(found_scripts)} legacy scripts found'
            elif found_scripts:
                status = 'warning'
                message = f'{len(found_scripts)}/{len(legacy_scripts)} legacy scripts found'
            else:
                status = 'error'
                message = 'No legacy scripts found'

            return {
                'status': status,
                'message': message,
                'details': {
                    'found': found_scripts,
                    'missing': missing_scripts,
                    'script_info': script_info,
                    'note': 'Legacy scripts will be gradually replaced by modular components'
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Legacy scripts check failed: {str(e)}',
                'details': {'exception': str(e)}
            }

    def full_system_check(self) -> Dict[str, Any]:
        """Run comprehensive system health assessment."""
        print("\U0001f3e5 Running comprehensive system health check...")

        # Run all individual checks
        checks = {
            'configuration': self.check_configuration(),
            'documents_folder': self.check_documents_folder(),
            'dependencies': self.check_dependencies(),
            'legacy_scripts': self.check_legacy_scripts()
        }

        # Determine overall system status
        statuses = [check['status'] for check in checks.values()]

        if all(status == 'healthy' for status in statuses):
            overall_status = 'healthy'
            overall_message = 'All system components healthy'
        elif any(status == 'error' for status in statuses):
            overall_status = 'error'
            error_components = [name for name, check in checks.items() if check['status'] == 'error']
            overall_message = f'System errors in: {", ".join(error_components)}'
        else:
            overall_status = 'warning'
            warning_components = [name for name, check in checks.items() if check['status'] == 'warning']
            overall_message = f'System warnings in: {", ".join(warning_components)}'

        # Add overall assessment
        checks['overall'] = {
            'status': overall_status,
            'message': overall_message,
            'details': {
                'total_checks': len(checks) - 1,
                'healthy_count': sum(1 for s in statuses if s == 'healthy'),
                'warning_count': sum(1 for s in statuses if s == 'warning'),
                'error_count': sum(1 for s in statuses if s == 'error')
            }
        }

        return checks

# Test the health checker
if __name__ == "__main__":
    print("\U0001f3e5 Testing Health Checker System...")
    print("=" * 50)

    try:
        # Initialize health checker
        health_checker = HealthChecker()

        # Run individual checks with detailed output
        individual_checks = [
            ('Configuration', health_checker.check_configuration),
            ('Documents Folder', health_checker.check_documents_folder),
            ('Dependencies', health_checker.check_dependencies),
            ('Legacy Scripts', health_checker.check_legacy_scripts)
        ]

        print("\nIndividual Health Checks:")
        for check_name, check_func in individual_checks:
            result = check_func()
            status_icon = {'healthy': '[OK]', 'warning': '[WARN]', 'error': '[ERR]'}.get(result['status'], '[?]')
            print(f"\n{status_icon} {check_name}:")
            print(f"  Status: {result['status']}")
            print(f"  Message: {result['message']}")

            # Show key details
            if 'details' in result:
                details = result['details']
                if isinstance(details, dict):
                    for key, value in list(details.items())[:3]:
                        print(f"  {key}: {value}")

        # Run full system check
        print("\nFull System Assessment:")
        full_results = health_checker.full_system_check()

        if 'overall' in full_results:
            overall = full_results['overall']
            status_icon = {'healthy': '[OK]', 'warning': '[WARN]', 'error': '[ERR]'}.get(overall['status'], '[?]')
            print(f"{status_icon} Overall Status: {overall['status'].upper()}")
            print(f"Message: {overall['message']}")

            if 'details' in overall:
                details = overall['details']
                print(f"Summary: {details['healthy_count']} healthy, {details['warning_count']} warnings, {details['error_count']} errors")

        print("\nHealth checker system working correctly")

    except Exception as e:
        print(f"Health checker test failed: {e}")
        import traceback
        traceback.print_exc()
