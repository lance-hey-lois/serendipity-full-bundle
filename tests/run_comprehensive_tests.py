#!/usr/bin/env python3
"""
Comprehensive Test Suite Runner
==============================

Main test runner for the quantum-enhanced serendipity engine.
Orchestrates all test categories and generates comprehensive reports.

Usage:
    python run_comprehensive_tests.py --all
    python run_comprehensive_tests.py --unit --integration
    python run_comprehensive_tests.py --performance --quantum-superiority
    python run_comprehensive_tests.py --quick
"""

import argparse
import subprocess
import sys
import os
import time
import json
from datetime import datetime
from typing import List, Dict, Any
import pytest

class TestSuiteRunner:
    """Orchestrate comprehensive test execution."""
    
    def __init__(self):
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def run_test_category(self, category: str, test_files: List[str], 
                         pytest_args: List[str] = None) -> Dict[str, Any]:
        """Run a category of tests."""
        print(f"\n{'='*60}")
        print(f"RUNNING {category.upper()} TESTS")
        print(f"{'='*60}")
        
        if pytest_args is None:
            pytest_args = ["-v", "-x", "--tb=short"]
        
        start_time = time.time()
        results = {"category": category, "files": test_files}
        
        for test_file in test_files:
            test_path = os.path.join(self.test_dir, test_file)
            
            if not os.path.exists(test_path):
                print(f"WARNING: Test file {test_file} not found, skipping...")
                continue
            
            print(f"\nRunning {test_file}...")
            
            # Run pytest programmatically
            try:
                result = pytest.main([test_path] + pytest_args)
                
                if result == 0:
                    status = "PASSED"
                elif result == 1:
                    status = "FAILED"
                elif result == 2:
                    status = "INTERRUPTED"
                elif result == 3:
                    status = "INTERNAL_ERROR"
                elif result == 4:
                    status = "USAGE_ERROR"
                elif result == 5:
                    status = "NO_TESTS"
                else:
                    status = f"UNKNOWN({result})"
                
                results[test_file] = {
                    "status": status,
                    "exit_code": result,
                    "success": result == 0
                }
                
                print(f"Result: {status}")
                
            except Exception as e:
                print(f"ERROR running {test_file}: {e}")
                results[test_file] = {
                    "status": "ERROR",
                    "exit_code": -1,
                    "success": False,
                    "error": str(e)
                }
        
        end_time = time.time()
        results["duration"] = end_time - start_time
        results["success"] = all(result.get("success", False) for result in results.values() 
                               if isinstance(result, dict) and "success" in result)
        
        print(f"\n{category} tests completed in {results['duration']:.2f}s - "
              f"{'PASSED' if results['success'] else 'FAILED'}")
        
        return results
    
    def run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests."""
        unit_tests = [
            "test_quantum_algorithms.py",
            "test_photonic_gbs.py",
            "test_serendipity_bandit.py",
            "test_multi_objective_scoring.py",
            "test_faiss_operations.py"
        ]
        
        return self.run_test_category("UNIT", unit_tests, ["-v", "--tb=short"])
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests."""
        integration_tests = [
            "test_integration.py"
        ]
        
        return self.run_test_category("INTEGRATION", integration_tests, 
                                    ["-v", "--tb=short", "-s"])
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance benchmarks."""
        performance_tests = [
            "test_performance_benchmarks.py"
        ]
        
        return self.run_test_category("PERFORMANCE", performance_tests, 
                                    ["-v", "--tb=short", "-s", "--durations=10"])
    
    def run_validation_tests(self) -> Dict[str, Any]:
        """Run continuous validation tests."""
        validation_tests = [
            "test_continuous_validation.py"
        ]
        
        return self.run_test_category("VALIDATION", validation_tests, 
                                    ["-v", "--tb=short", "-s"])
    
    def run_quantum_superiority_tests(self) -> Dict[str, Any]:
        """Run quantum superiority analysis."""
        superiority_tests = [
            "test_quantum_superiority.py"
        ]
        
        return self.run_test_category("QUANTUM_SUPERIORITY", superiority_tests, 
                                    ["-v", "--tb=short", "-s"])
    
    def run_quick_tests(self) -> Dict[str, Any]:
        """Run quick smoke tests."""
        quick_tests = [
            "test_quantum_algorithms.py::TestQuantumKernelMethods::test_pca_compress_basic_functionality",
            "test_photonic_gbs.py::TestGBSCore::test_have_photonics_detection",
            "test_serendipity_bandit.py::TestSerendipityBins::test_make_bins_basic_functionality",
            "test_multi_objective_scoring.py::TestCosineSimilarity::test_cos_basic_functionality",
            "test_faiss_operations.py::TestFAISSDetection::test_have_faiss_detection"
        ]
        
        print(f"\n{'='*60}")
        print(f"RUNNING QUICK SMOKE TESTS")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Run selected tests
        result = pytest.main(["-v"] + [os.path.join(self.test_dir, test) for test in quick_tests])
        
        end_time = time.time()
        
        return {
            "category": "QUICK",
            "tests": quick_tests,
            "duration": end_time - start_time,
            "success": result == 0,
            "exit_code": result
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_duration = (self.end_time - self.start_time) if self.end_time and self.start_time else 0
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_duration": total_duration,
            "results": self.results,
            "summary": {
                "categories_run": len(self.results),
                "categories_passed": sum(1 for r in self.results.values() if r.get("success", False)),
                "overall_success": all(r.get("success", False) for r in self.results.values())
            }
        }
        
        # Count individual test files
        total_files = 0
        passed_files = 0
        
        for category_result in self.results.values():
            if isinstance(category_result, dict):
                for key, value in category_result.items():
                    if key.endswith(".py") and isinstance(value, dict):
                        total_files += 1
                        if value.get("success", False):
                            passed_files += 1
        
        report["summary"]["total_test_files"] = total_files
        report["summary"]["passed_test_files"] = passed_files
        
        return report
    
    def print_summary(self, report: Dict[str, Any]):
        """Print test summary."""
        print(f"\n{'='*60}")
        print(f"TEST EXECUTION SUMMARY")
        print(f"{'='*60}")
        
        summary = report["summary"]
        
        print(f"Timestamp: {report['timestamp']}")
        print(f"Total Duration: {report['total_duration']:.2f}s")
        print(f"Categories Run: {summary['categories_run']}")
        print(f"Categories Passed: {summary['categories_passed']}")
        print(f"Test Files Run: {summary['total_test_files']}")
        print(f"Test Files Passed: {summary['passed_test_files']}")
        
        status = "PASSED" if summary["overall_success"] else "FAILED"
        print(f"Overall Status: {status}")
        
        print(f"\nDetailed Results:")
        for category, result in report["results"].items():
            if isinstance(result, dict):
                category_status = "PASSED" if result.get("success", False) else "FAILED"
                duration = result.get("duration", 0)
                print(f"  {category:20s}: {category_status:8s} ({duration:6.2f}s)")
        
        print(f"{'='*60}")
        
        return summary["overall_success"]

def main():
    parser = argparse.ArgumentParser(description="Run comprehensive test suite for quantum serendipity engine")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests") 
    parser.add_argument("--performance", action="store_true", help="Run performance benchmarks")
    parser.add_argument("--validation", action="store_true", help="Run validation tests")
    parser.add_argument("--quantum-superiority", action="store_true", help="Run quantum superiority analysis")
    parser.add_argument("--quick", action="store_true", help="Run quick smoke tests only")
    parser.add_argument("--report", type=str, help="Save report to JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # If no specific tests specified, run all
    if not any([args.unit, args.integration, args.performance, args.validation, 
               args.quantum_superiority, args.quick]):
        args.all = True
    
    runner = TestSuiteRunner()
    runner.start_time = time.time()
    
    try:
        if args.quick:
            runner.results["quick"] = runner.run_quick_tests()
        else:
            if args.all or args.unit:
                runner.results["unit"] = runner.run_unit_tests()
            
            if args.all or args.integration:
                runner.results["integration"] = runner.run_integration_tests()
            
            if args.all or args.performance:
                runner.results["performance"] = runner.run_performance_tests()
            
            if args.all or args.validation:
                runner.results["validation"] = runner.run_validation_tests()
            
            if args.all or args.quantum_superiority:
                runner.results["quantum_superiority"] = runner.run_quantum_superiority_tests()
    
    except KeyboardInterrupt:
        print("\n\nTest execution interrupted by user.")
        return 1
    
    except Exception as e:
        print(f"\n\nUnexpected error during test execution: {e}")
        return 1
    
    finally:
        runner.end_time = time.time()
    
    # Generate and display report
    report = runner.generate_report()
    success = runner.print_summary(report)
    
    # Save report if requested
    if args.report:
        with open(args.report, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nDetailed report saved to: {args.report}")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)