#!/usr/bin/env python3
"""
Serendipity Engine UI - System Verification
==========================================

Comprehensive verification script for MongoDB, dependencies, and services.
This script tests all critical components before system startup.
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import importlib.util

def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)

def print_status(item: str, status: bool, details: str = ""):
    """Print formatted status line"""
    symbol = "✅" if status else "❌"
    print(f"{symbol} {item}")
    if details and not status:
        print(f"   Details: {details}")

class SystemVerifier:
    def __init__(self):
        self.results = {}
        self.errors = []
        
    def verify_python_version(self) -> bool:
        """Verify Python version is compatible"""
        print_section("Python Version Check")
        
        version = sys.version_info
        required_major, required_minor = 3, 8
        
        compatible = version.major >= required_major and version.minor >= required_minor
        
        print_status(
            f"Python {version.major}.{version.minor}.{version.micro}",
            compatible,
            f"Requires Python {required_major}.{required_minor}+"
        )
        
        self.results['python_version'] = compatible
        return compatible
    
    def verify_node_npm(self) -> bool:
        """Verify Node.js and npm are available"""
        print_section("Node.js and npm Check")
        
        try:
            # Check Node.js
            node_result = subprocess.run(['node', '--version'], 
                                       capture_output=True, text=True, timeout=10)
            node_ok = node_result.returncode == 0
            node_version = node_result.stdout.strip() if node_ok else "Not found"
            
            # Check npm
            npm_result = subprocess.run(['npm', '--version'], 
                                      capture_output=True, text=True, timeout=10)
            npm_ok = npm_result.returncode == 0
            npm_version = npm_result.stdout.strip() if npm_ok else "Not found"
            
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            node_ok = npm_ok = False
            node_version = npm_version = f"Error: {e}"
        
        print_status(f"Node.js {node_version}", node_ok)
        print_status(f"npm {npm_version}", npm_ok)
        
        self.results['node_npm'] = node_ok and npm_ok
        return node_ok and npm_ok
    
    def verify_mongodb_connection(self) -> bool:
        """Test MongoDB connection using environment credentials"""
        print_section("MongoDB Connection Test")
        
        # Load environment variables
        env_path = Path(".env")
        if env_path.exists():
            from dotenv import load_dotenv
            load_dotenv(env_path)
        
        mongodb_uri = os.getenv("MONGODB_URI")
        db_name = os.getenv("DB_NAME", "MagicCRM")
        
        if not mongodb_uri:
            print_status("MongoDB URI", False, "MONGODB_URI not found in environment")
            self.results['mongodb'] = False
            return False
        
        try:
            from pymongo import MongoClient
            from pymongo.errors import ConnectionFailure, ConfigurationError
            
            # Test connection with timeout
            client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
            
            # Actually test the connection
            client.admin.command('ping')
            
            # Test database access
            db = client[db_name]
            collections = db.list_collection_names()
            
            print_status(f"MongoDB Connection", True)
            print_status(f"Database '{db_name}' access", True)
            print_status(f"Collections found: {len(collections)}", True)
            
            # Check for required collections
            required_collections = ['public_profiles']
            for collection in required_collections:
                exists = collection in collections
                print_status(f"Collection '{collection}'", exists)
                if exists:
                    count = db[collection].count_documents({})
                    print(f"   Documents: {count}")
            
            client.close()
            self.results['mongodb'] = True
            return True
            
        except ImportError as e:
            print_status("pymongo module", False, str(e))
            self.results['mongodb'] = False
            return False
        except (ConnectionFailure, ConfigurationError) as e:
            print_status("MongoDB Connection", False, str(e))
            self.results['mongodb'] = False
            return False
        except Exception as e:
            print_status("MongoDB Test", False, f"Unexpected error: {e}")
            self.results['mongodb'] = False
            return False
    
    def verify_python_dependencies(self) -> bool:
        """Check if Python dependencies are installed"""
        print_section("Python Dependencies Check")
        
        # Read requirements from files
        requirements_files = [
            'requirements.txt',
            'backend/requirements.txt',
            'requirements-quantum-api.txt'
        ]
        
        all_requirements = set()
        for req_file in requirements_files:
            req_path = Path(req_file)
            if req_path.exists():
                with open(req_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # Extract package name (before any version specifiers)
                            pkg_name = line.split('>=')[0].split('==')[0].split('<')[0].split('>')[0]
                            all_requirements.add(pkg_name.strip())
        
        # Core requirements to check
        core_requirements = [
            'fastapi', 'uvicorn', 'pymongo', 'python-dotenv', 'openai',
            'numpy', 'pydantic', 'strawberry-graphql', 'motor'
        ]
        
        # Combine with found requirements
        to_check = list(all_requirements.union(set(core_requirements)))
        
        missing = []
        available = []
        
        for package in to_check:
            try:
                spec = importlib.util.find_spec(package.replace('-', '_'))
                if spec is not None:
                    available.append(package)
                    print_status(f"Package '{package}'", True)
                else:
                    missing.append(package)
                    print_status(f"Package '{package}'", False, "Not installed")
            except (ImportError, ModuleNotFoundError, ValueError):
                missing.append(package)
                print_status(f"Package '{package}'", False, "Import error")
        
        success = len(missing) == 0
        
        if missing:
            print(f"\n❌ Missing packages: {', '.join(missing)}")
            print("Run: pip install -r requirements.txt")
        
        self.results['python_deps'] = success
        return success
    
    def verify_react_dependencies(self) -> bool:
        """Check React app dependencies"""
        print_section("React Dependencies Check")
        
        react_dir = Path("quantum-discovery-react")
        if not react_dir.exists():
            print_status("React directory", False, "quantum-discovery-react not found")
            self.results['react_deps'] = False
            return False
        
        package_json = react_dir / "package.json"
        if not package_json.exists():
            print_status("package.json", False, "File not found")
            self.results['react_deps'] = False
            return False
        
        node_modules = react_dir / "node_modules"
        installed = node_modules.exists()
        
        print_status("package.json", True)
        print_status("node_modules", installed, "Run 'npm install' if missing")
        
        if installed:
            # Check for key dependencies
            key_deps = ['react', '@types/react', 'typescript', 'react-scripts']
            for dep in key_deps:
                dep_path = node_modules / dep
                exists = dep_path.exists()
                print_status(f"Dependency '{dep}'", exists)
        
        self.results['react_deps'] = installed
        return installed
    
    def verify_environment_files(self) -> bool:
        """Check environment configuration"""
        print_section("Environment Configuration Check")
        
        env_files = ['.env', 'config/environment.env']
        env_found = False
        
        for env_file in env_files:
            env_path = Path(env_file)
            exists = env_path.exists()
            print_status(f"Environment file '{env_file}'", exists)
            if exists:
                env_found = True
                
                # Check for required variables
                with open(env_path, 'r') as f:
                    content = f.read()
                    
                required_vars = ['MONGODB_URI', 'OPENAI_API_KEY', 'DB_NAME']
                for var in required_vars:
                    has_var = var in content and f"{var}=" in content
                    print_status(f"  Variable '{var}'", has_var)
        
        self.results['environment'] = env_found
        return env_found
    
    def test_api_imports(self) -> bool:
        """Test if API modules can be imported"""
        print_section("API Module Import Test")
        
        api_files = [
            ('serendipity_api.py', 'serendipity_api'),
            ('quantum_api.py', 'quantum_api'), 
            ('backend/main.py', 'backend.main')
        ]
        
        all_imports_ok = True
        
        for file_path, module_name in api_files:
            if Path(file_path).exists():
                try:
                    # Add current directory to path for imports
                    if str(Path.cwd()) not in sys.path:
                        sys.path.insert(0, str(Path.cwd()))
                    
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    if spec and spec.loader:
                        print_status(f"Module '{file_path}' importable", True)
                    else:
                        print_status(f"Module '{file_path}' importable", False, "Spec creation failed")
                        all_imports_ok = False
                        
                except Exception as e:
                    print_status(f"Module '{file_path}' importable", False, str(e))
                    all_imports_ok = False
            else:
                print_status(f"File '{file_path}' exists", False)
                all_imports_ok = False
        
        self.results['api_imports'] = all_imports_ok
        return all_imports_ok
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive verification report"""
        print_section("System Verification Summary")
        
        all_good = all(self.results.values())
        
        print(f"Overall Status: {'✅ PASS' if all_good else '❌ FAIL'}")
        print(f"\nDetailed Results:")
        
        for check, result in self.results.items():
            print_status(check.replace('_', ' ').title(), result)
        
        return {
            'overall_status': 'PASS' if all_good else 'FAIL',
            'checks': self.results,
            'ready_for_startup': all_good,
            'timestamp': time.time()
        }

def main():
    """Run complete system verification"""
    print("Serendipity Engine UI - System Verification")
    print("=" * 60)
    
    verifier = SystemVerifier()
    
    # Run all verification checks
    checks = [
        verifier.verify_python_version,
        verifier.verify_node_npm,
        verifier.verify_environment_files,
        verifier.verify_mongodb_connection,
        verifier.verify_python_dependencies,
        verifier.verify_react_dependencies,
        verifier.test_api_imports
    ]
    
    for check in checks:
        try:
            check()
        except Exception as e:
            print(f"❌ Check failed with error: {e}")
            verifier.results[check.__name__] = False
    
    # Generate final report
    report = verifier.generate_report()
    
    # Save report
    report_path = Path("scripts/verification_report.json")
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to: {report_path}")
    
    return report['ready_for_startup']

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)