#!/usr/bin/env python3
"""
Performance Validation Script

This script validates the performance improvement achieved by switching
from bitflip_float32 (slow) to bitflip_float32_optimized (fast).
"""

import time

import torch

from seu_injection.bitops.float32 import bitflip_float32, bitflip_float32_optimized


def benchmark_bitflip_functions():
    """Compare performance of old vs new bitflip functions."""
    print("🔬 PERFORMANCE VALIDATION - BITFLIP OPERATIONS")
    print("=" * 60)

    # Create test data
    test_tensor = torch.randn(1000, dtype=torch.float32)
    bit_position = 15  # Middle mantissa bit

    # Benchmark old function (slow)
    print("📊 Testing bitflip_float32 (OLD)...")
    start_time = time.time()
    for _ in range(100):  # Fewer iterations for old function
        _ = bitflip_float32(test_tensor.clone(), bit_position)
    old_time = time.time() - start_time
    old_avg = old_time / 100 * 1000  # ms per operation

    # Benchmark new function (optimized)
    print("⚡ Testing bitflip_float32_optimized (NEW)...")
    start_time = time.time()
    for _ in range(1000):  # More iterations for fair comparison
        _ = bitflip_float32_optimized(test_tensor.clone(), bit_position)
    new_time = time.time() - start_time
    new_avg = new_time / 1000 * 1000  # ms per operation

    # Calculate improvement
    speedup = old_avg / max(new_avg, 0.001)  # Avoid division by zero

    print("\n📈 RESULTS:")
    print(f"   Old function: {old_avg:.2f} ms per operation")
    print(f"   New function: {new_avg:.2f} ms per operation")
    print(f"   Speedup: {speedup:.1f}x faster")
    print(f"   Time saved: {((old_avg - new_avg) / old_avg * 100):.1f}%")

    print("\n🎯 IMPACT ON REAL USAGE:")
    # Estimate for ResNet-18 scale (11M parameters)
    resnet_params = 11_000_000
    old_total = resnet_params * old_avg / 1000  # seconds
    new_total = resnet_params * new_avg / 1000  # seconds

    print(f"   ResNet-18 full injection (old): {old_total / 60:.1f} minutes")
    print(f"   ResNet-18 full injection (new): {new_total / 60:.1f} minutes")
    print(
        f"   Time saved per full analysis: {(old_total - new_total) / 60:.1f} minutes"
    )

    return speedup > 10  # Should be at least 10x faster


def main():
    """Main validation routine."""
    print("Starting performance validation...\n")

    success = benchmark_bitflip_functions()

    print("\n" + "=" * 60)
    if success:
        print("✅ VALIDATION SUCCESSFUL - Performance improvement confirmed!")
        print("🚀 The optimization is working as expected.")
    else:
        print("❌ VALIDATION FAILED - Performance improvement not achieved")
        print("🔍 Check that bitflip_float32_optimized is being used correctly")

    return success


if __name__ == "__main__":
    main()
