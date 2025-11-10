#!/usr/bin/env python3
"""
Pipeline validation for basic_cnn_robustness.py example.

This script can be run directly in CI/CD to validate that the
basic_cnn_robustness.py example works correctly. It uses minimal
parameters for fast execution while still testing the complete pipeline.
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def run_example_with_timeout(timeout_seconds=300):
    """
    Run the basic_cnn_robustness.py example with a timeout.

    Args:
        timeout_seconds: Maximum time to allow the example to run

    Returns:
        bool: True if example completed successfully, False otherwise
    """
    examples_dir = Path(__file__).parent.parent.parent / "examples"
    example_path = examples_dir / "basic_cnn_robustness.py"

    if not example_path.exists():
        print(f"FAILED: Example not found: {example_path}")
        return False

    print(f"Running example: {example_path}")
    print(f"Timeout: {timeout_seconds} seconds")

    try:
        # Run the example with UV
        result = subprocess.run(
            ["uv", "run", "python", str(example_path)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=examples_dir.parent,
        )

        if result.returncode == 0:
            print("SUCCESS: Example completed successfully!")
            print("STDOUT:", result.stdout[-500:])  # Show last 500 chars
            return True
        else:
            print(f"FAILED: Example failed with return code: {result.returncode}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print(f"FAILED: Example timed out after {timeout_seconds} seconds")
        print("This suggests the example may be too slow for CI/CD")
        return False
    except subprocess.CalledProcessError as e:
        print(f"FAILED: Example execution error: {e}")
        return False
    except FileNotFoundError:
        print("FAILED: UV not found. Please ensure UV is installed and in PATH")
        return False


def check_example_dependencies():
    """Check if example dependencies are available."""
    print("INFO: Checking example dependencies...")

    try:
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-c",
                "import torch, numpy, sklearn, matplotlib, seu_injection; print('SUCCESS: All dependencies available')",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            print("SUCCESS: All dependencies are available")
            return True
        else:
            print("FAILED: Some dependencies missing:")
            print("STDERR:", result.stderr)
            return False

    except (
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        FileNotFoundError,
    ) as e:
        print(f"FAILED: Dependency check failed: {e}")
        return False


def check_example_syntax():
    """Check if example has valid Python syntax."""
    examples_dir = Path(__file__).parent.parent.parent / "examples"
    example_path = examples_dir / "basic_cnn_robustness.py"

    print("INFO: Checking example syntax...")

    try:
        result = subprocess.run(
            ["uv", "run", "python", "-m", "py_compile", str(example_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            print("SUCCESS: Example syntax is valid")
            return True
        else:
            print("FAILED: Example has syntax errors:")
            print("STDERR:", result.stderr)
            return False

    except (
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        FileNotFoundError,
    ) as e:
        print(f"FAILED: Syntax check failed: {e}")
        return False


def main():
    """Main pipeline validation function."""
    print("PIPELINE VALIDATION: basic_cnn_robustness.py")
    print("=" * 60)

    success = True

    # 1. Check syntax
    if not check_example_syntax():
        success = False

    # 2. Check dependencies
    if not check_example_dependencies():
        success = False

    # 3. Run example with reasonable timeout (5 minutes)
    if success:
        if not run_example_with_timeout(timeout_seconds=300):
            success = False

    # 4. Summary
    print("\n" + "=" * 60)
    if success:
        print("SUCCESS: PIPELINE VALIDATION PASSED")
        print("basic_cnn_robustness.py example is ready for production!")
    else:
        print("FAILED: PIPELINE VALIDATION FAILED")
        print("basic_cnn_robustness.py example needs fixes before production")
    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
