import pytest
import sys
import os

# Add the parent directory to sys.path so we can import from framework/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from framework.bitflip import bitflip_float32

def test_bitflip_float32_basic():
    # Flipping bit 0 of float32(1.0) should result in -1.0 (IEEE-754 representation)
    result = bitflip_float32(1.0, 0)
    assert result == -1.0, f"Expected -1.0, got {result}"
