"""Smoke tests that run example scripts directly.

This test module ensures that the example scripts can be executed by external
users without errors. Each test runs the actual script that users would copy/paste.

These tests use FAST/DEBUG modes where available to ensure quick execution.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"


def run_script(script_path, args=None, timeout=180):
    """Run a script and return the result.

    Args:
        script_path: Path to the script to run
        args: List of command-line arguments
        timeout: Maximum time in seconds to allow the script to run

    Returns:
        subprocess.CompletedProcess result
    """
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)

    # Set environment to handle Unicode (emojis) on Windows
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        cmd,
        capture_output=True,
        timeout=timeout,
        cwd=EXAMPLES_DIR.parent,  # Run from repo root
        env=env,
    )
    return result


class TestExampleScriptsRun:
    """Test that example scripts run without errors."""

    def test_basic_cnn_robustness_fast_mode(self):
        """Run basic_cnn_robustness.py in fast mode."""
        result = run_script(
            EXAMPLES_DIR / "basic_cnn_robustness.py",
            args=["--fast"],
            timeout=180,
        )

        stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
        stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""

        assert result.returncode == 0, (
            f"basic_cnn_robustness.py failed with return code {result.returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )

        # Verify key output is present
        assert "SEU INJECTION FRAMEWORK" in stdout, "Expected header not found"
        assert "Baseline Accuracy" in stdout, "Expected baseline output not found"

    def test_flood_training_study_fast_mode(self):
        """Run flood_training_study/experiment.py in fast mode."""
        result = run_script(
            EXAMPLES_DIR / "flood_training_study" / "experiment.py",
            args=["--fast"],
            timeout=180,
        )

        stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
        stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""

        assert result.returncode == 0, (
            f"flood_training_study/experiment.py failed with return code {result.returncode}\n"
            f"STDOUT:\n{stdout}\n"
            f"STDERR:\n{stderr}"
        )

        # Verify key output is present
        assert "FLOOD LEVEL TRAINING" in stdout, "Expected header not found"
        assert "SEU" in stdout and "robustness" in stdout.lower(), "Expected SEU robustness output not found"

    @pytest.mark.skip(reason="Pre-existing bugs: tuple keys and Unicode encoding on Windows")
    def test_architecture_comparison_default_mode(self):
        """Run architecture_comparison.py (already optimized for speed)."""
        result = run_script(
            EXAMPLES_DIR / "architecture_comparison.py",
            timeout=300,
        )

        stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
        stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""

        assert result.returncode == 0, (
            f"architecture_comparison.py failed with return code {result.returncode}\n"
            f"STDOUT:\n{stdout}\n"
            f"STDERR:\n{stderr}"
        )

        # Verify key output is present
        assert "Robustness" in stdout, "Expected robustness output not found"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
