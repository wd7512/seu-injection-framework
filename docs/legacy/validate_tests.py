#!/usr/bin/env python3
"""
Pre-installation validation script.

This script validates the testing framework structure and configuration
without requiring dependencies to be installed yet.
"""

import sys
import os
from pathlib import Path
import ast

def check_file_exists(filepath, description):
    """Check if a file exists and report."""
    if filepath.exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} - MISSING")
        return False

def validate_python_syntax(filepath):
    """Validate Python file syntax without importing."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        ast.parse(content)
        print(f"✅ Syntax valid: {filepath.name}")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in {filepath}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error checking {filepath}: {e}")
        return False

def main():
    """Validate testing framework setup."""
    print("🔍 Validating testing framework setup...")
    print("=" * 60)
    
    framework_root = Path(__file__).parent
    all_good = True
    
    # Check core test files
    test_files = [
        (framework_root / "tests" / "conftest.py", "Test configuration"),
        (framework_root / "tests" / "test_bitflip.py", "Bitflip unit tests"),
        (framework_root / "tests" / "test_injector.py", "Injector unit tests"), 
        (framework_root / "tests" / "test_criterion.py", "Criterion unit tests"),
        (framework_root / "tests" / "integration" / "test_workflows.py", "Integration tests"),
        (framework_root / "tests" / "smoke" / "test_basic_functionality.py", "Smoke tests"),
        (framework_root / "run_tests.py", "Test runner script"),
        (framework_root / "pyproject.toml", "Project configuration"),
    ]
    
    print("\n📁 File Structure Validation:")
    for filepath, description in test_files:
        if not check_file_exists(filepath, description):
            all_good = False
    
    # Check Python syntax  
    print("\n🐍 Python Syntax Validation:")
    python_files = [f[0] for f in test_files if f[0].suffix == '.py']
    
    for filepath in python_files:
        if filepath.exists():
            if not validate_python_syntax(filepath):
                all_good = False
    
    # Check pyproject.toml syntax
    print("\n📋 Configuration Validation:")
    pyproject_file = framework_root / "pyproject.toml"
    if pyproject_file.exists():
        try:
            # Try to read as text first
            with open(pyproject_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic validation - should contain key sections
            required_sections = ['[project]', '[build-system]', '[tool.pytest.ini_options]']
            missing_sections = []
            
            for section in required_sections:
                if section not in content:
                    missing_sections.append(section)
            
            if missing_sections:
                print(f"❌ pyproject.toml missing sections: {missing_sections}")
                all_good = False
            else:
                print("✅ pyproject.toml contains required sections")
                
        except Exception as e:
            print(f"❌ Error reading pyproject.toml: {e}")
            all_good = False
    
    # Check test organization
    print("\n🧪 Test Organization Validation:")
    
    # Count tests by category
    test_counts = {
        "Unit tests": len(list((framework_root / "tests").glob("test_*.py"))),
        "Integration tests": len(list((framework_root / "tests" / "integration").glob("*.py"))),
        "Smoke tests": len(list((framework_root / "tests" / "smoke").glob("*.py"))),
    }
    
    for category, count in test_counts.items():
        if count > 0:
            print(f"✅ {category}: {count} files")
        else:
            print(f"⚠️  {category}: {count} files")
    
    # Summary
    print("\n" + "=" * 60)
    if all_good:
        print("✅ TESTING FRAMEWORK VALIDATION PASSED!")
        print("\n🚀 Next steps:")
        print("1. Install UV: curl -LsSf https://astral.sh/uv/install.sh | sh")
        print("2. Install dependencies: uv sync --all-extras")
        print("3. Run smoke tests: python run_tests.py smoke")
        print("4. Run full test suite: python run_tests.py all")
    else:
        print("❌ TESTING FRAMEWORK VALIDATION FAILED!")
        print("Please fix the issues above before proceeding.")
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())