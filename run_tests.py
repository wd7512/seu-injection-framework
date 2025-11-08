#!/usr/bin/env python3
"""
Test runner script for the SEU Injection Framework.

This script provides convenient commands to run different test suites:
- Smoke tests: Quick validation of basic functionality
- Unit tests: Detailed testing of individual components  
- Integration tests: End-to-end workflow testing
- All tests: Complete test suite

Usage:
    python run_tests.py smoke      # Quick smoke tests
    python run_tests.py unit       # Unit tests only
    python run_tests.py integration # Integration tests only  
    python run_tests.py all        # All tests
    python run_tests.py --help     # Show help
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

# Add framework to path
framework_root = Path(__file__).parent
sys.path.insert(0, str(framework_root))


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=framework_root)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {description}")
        print(f"Exit code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        
        # Check if this is a coverage failure
        if "--cov-fail-under" in ' '.join(cmd) and "coverage" in description.lower():
            output_text = e.stdout + e.stderr
            if "TOTAL" in output_text and "%" in output_text:
                print(f"\nCOVERAGE REQUIREMENT NOT MET!")
                print(f"Current coverage is below the required 80% threshold.")
                print(f"To fix coverage issues:")
                print(f"   1. Run with verbose coverage: uv run pytest --cov=framework --cov-report=term-missing -v")
                print(f"   2. Check htmlcov/index.html for detailed coverage report")
                print(f"   3. Add tests for uncovered code paths")
                print(f"   4. Ensure all critical functions have test coverage")
                print(f"WARNING: Maintaining 80% coverage is required for code quality standards.")
        
        return False
    except FileNotFoundError:
        print(f"ERROR: Command not found. Make sure UV is installed and in PATH.")
        print(f"Try: curl -LsSf https://astral.sh/uv/install.sh | sh")
        return False


def run_smoke_tests():
    """Run smoke tests for quick validation."""
    print("Running smoke tests (quick validation)...")
    
    # First try direct Python execution via UV
    smoke_test_file = framework_root / "tests" / "smoke" / "test_basic_functionality.py"
    if smoke_test_file.exists():
        cmd = ["uv", "run", "python", str(smoke_test_file)]
        success = run_command(cmd, "Direct smoke test execution (UV)")
        if success:
            return True
    
    # Fallback to pytest via UV
    cmd = ["uv", "run", "pytest", "tests/smoke/", "-v"]
    return run_command(cmd, "Smoke tests via UV pytest")


def run_unit_tests():
    """Run unit tests."""
    print("Running unit tests...")
    cmd = ["uv", "run", "pytest", "tests/test_bitflip.py", "tests/test_criterion.py", "tests/test_injector.py", "-v", "--tb=short"]
    return run_command(cmd, "Unit tests")


def run_integration_tests():
    """Run integration tests."""
    print("Running integration tests...")
    cmd = ["uv", "run", "pytest", "tests/integration/", "-v", "--tb=short"]
    return run_command(cmd, "Integration tests")


def run_all_tests():
    """Run all tests with coverage."""
    print("Running complete test suite with coverage...")
    cmd = [
        "uv", "run", "pytest", 
        "tests/", 
        "-v", 
        "--cov=src/seu_injection", 
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "--cov-fail-under=80",
        "--tb=short"
    ]
    return run_command(cmd, "Complete test suite with coverage")


def check_dependencies():
    """Check if all required dependencies are available."""
    print("Checking test dependencies via UV environment...")
    
    # Check if UV environment exists and has dependencies
    try:
        result = subprocess.run(
            ["uv", "run", "python", "-c", 
             "import pytest, torch, numpy, pandas, sklearn; print('OK All dependencies available!')"],
            capture_output=True, 
            text=True, 
            cwd=framework_root,
            timeout=30
        )
        
        if result.returncode == 0:
            print("OK All dependencies available in UV environment!")
            return True
        else:
            print("ERROR Some dependencies missing from UV environment")
            print("STDERR:", result.stderr)
            print("Try: uv sync --all-extras")
            return False
            
    except subprocess.TimeoutExpired:
        print("ERROR: Dependency check timed out")
        return False
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Dependency check failed: {e}")
        return False
    except FileNotFoundError:
        print("ERROR: UV not found. Please install UV first.")
        print("Install with: curl -LsSf https://astral.sh/uv/install.sh | sh")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run tests for SEU Injection Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'test_type', 
        choices=['smoke', 'unit', 'integration', 'all', 'deps'],
        help='Type of tests to run'
    )
    
    parser.add_argument(
        '--no-deps-check',
        action='store_true',
        help='Skip dependency checking'
    )
    
    args = parser.parse_args()
    
    # Check dependencies first (unless skipped)
    if not args.no_deps_check and args.test_type != 'deps':
        if not check_dependencies():
            print("\nERROR: Dependency check failed. Fix dependencies and try again.")
            return 1
    
    # Run requested tests
    success = False
    
    if args.test_type == 'deps':
        success = check_dependencies()
    elif args.test_type == 'smoke':
        success = run_smoke_tests()
    elif args.test_type == 'unit':
        success = run_unit_tests()  
    elif args.test_type == 'integration':
        success = run_integration_tests()
    elif args.test_type == 'all':
        success = run_all_tests()
    
    # Summary
    print(f"\n{'='*60}")
    if success:
        print(f"SUCCESS {args.test_type.upper()} TESTS PASSED!")
    else:
        print(f"FAILED: {args.test_type.upper()} TESTS FAILED!")
    print(f"{'='*60}")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())