[0m
> build · big-pickle
[0m
[0m→ [0mRead tests/unit_tests/test_bitflip_hardcoded.py
[0m✱ [0mGlob "src/seu_injection/bitops.py"[90m 0 matches[0m
[0m✱ [0mGrep "bitflip_float32"[90m in src · 19 matches[0m
[0m→ [0mRead src/seu_injection/bitops/float32.py
[0m→ [0mRead src/seu_injection/bitops/float32_legacy.py
[0m→ [0mRead src/seu_injection/bitops/utils.py
[0m→ [0mRead src/seu_injection/bitops/__init__.py
[0m✱ [0mGlob "tests/unit_tests/test_bitflip*.py"[90m 3 matches[0m
[0m✱ [0mGrep "NaN|nan|denormal|subnormal|inplace"[90m in tests · 25 matches[0m
Here is the full review of `tests/unit_tests/test_bitflip_hardcoded.py`:

---

## Edge Case Coverage

### CRITICAL: NaN is entirely untested
No test flips bits **on** a NaN value, nor asserts creating NaN from a normal float. Flipping exponent bits on certain patterns yields NaN, and the legacy string-based path (`float32_to_binary`/`binary_to_float32` in `bitops/utils.py`) uses `struct.pack("!f", ...)` which may behave unexpectedly on NaN payloads. This is a real gap — if NaN-to-NaN bitflips silently corrupt, the SEU injector could produce undetectably wrong results.

### CRITICAL: Infinity as input is untested
`test_exponent_bit_flip_known_value` (L114-130) tests creating `inf` from `1.0`, but no test flips any bit **on** an infinity value (positive or negative). The legacy string-based path may behave differently from the optimized bit-mask path here since infinity has no valid mantissa bits.

### IMPORTANT: Denormal/subnormal numbers are untested
No test uses a denormalized float32 (< 1.175×10⁻³⁸). Flipping bits in the denormal range behaves differently (exponent = 0, implicit leading bit = 0). The code paths should be validated since the SEU injector targets neural network weights which may include near-zero values.

### NICE-TO-HAVE: Negative zero is barely tested
`test_zero_sign_bit_flip` (L149-162) flips bit 0 on `+0.0` → `-0.0` and back, but never flips a **non-sign** bit on `-0.0`. Negative zero's bit pattern (`0x80000000`) has a 1 in the sign bit and zeros elsewhere — flipping exponent or mantissa bits on it behaves differently than on positive zero.

---

## Assertion Quality

### IMPORTANT: Loose tolerance on LSB flips (L176, L233)
- `test_specific_small_value_flip` L176: `abs(result - 0.5) < 0.0001` for a bit-31 flip on 0.5. The actual ULP at 0.5 is 2⁻²⁴ ≈ 6×10⁻⁸. The tolerance is **~1600× too loose** — a wildly wrong result would still pass.
- `test_sequence_of_specific_flips` L233: `abs(step3 - 10.0) < 0.01` for a bit-31 flip on 10.0. The actual ULP is ~9.5×10⁻⁷. Tolerance is **~10000× too loose**.

Compare with `test_mantissa_lsb_flip_known_value` (L147) which correctly uses `2 * expected_diff` — this is the right pattern.

### NICE-TO-HAVE: Sloppy tolerance on large float (L186)
`test_large_value_flip` L186 uses `abs(result + 123456.78) < 0.01` for a **sign-bit flip**. The result should be exactly `-123456.78` — `==` would be appropriate and stricter.

---

## Precision Tolerances

### NICE-TO-HAVE: Inconsistent tolerance constants across tests
The file mixes `1e-5` (L202), `1e-6` (L251-257), `0.01` (L186, L233), and `0.0001` (L176, L349) without rationale. Each should be tied to the actual ULP at the test value for that bit position.

---

## Cross-Implementation Checks

### IMPORTANT: `test_all_implementations_consistent_hardcoded` only tests bit 0 (L237-243)
The cross-implementation consistency check only tests the sign bit. Exponent and mantissa bit flips are never cross-checked across all three implementations. If `bitflip_float32_optimized` and `bitflip_float32_fast` share a bug in the optimized path, or the legacy path diverges on exponent/mantissa bits, this test suite won't catch it.

### NICE-TO-HAVE: Many per-implementation tests only use one variant
- `test_mantissa_lsb_flip_known_value` (L132) and `test_exponent_bit_flip_known_value` (L114) only test `bitflip_float32` (legacy)
- `test_specific_bit_positions_on_one` (L325) only tests `bitflip_float32` (legacy) on 4 bit positions — all three implementations should agree

---

## Missing Test Dimensions

### CRITICAL: No `inplace=True` coverage
Two of the three implementations accept `inplace=True` and follow different code paths. Not tested at all in this file. (Covered in `test_bitflip.py` and `test_bitflip_optimized.py`, but this file is the "hardcoded" suite and should include at least one inplace check.)

### IMPORTANT: No ValueError for invalid bit positions
No test verifies that `bit_i` outside [0, 31] raises `ValueError`. If a caller passes e.g. `-1` or `32`, the behavior should be tested — especially since the SEU injector might programmatically generate bit positions.

### IMPORTANT: No exponent/mantissa bit flips cross-checked on arrays
`test_array_sign_bit_flips_all_elements` (L97) only tests bit 0 on arrays. No array test flips an exponent or mantissa bit across all elements.

### NICE-TO-HAVE: Bit positions 2–8 and 10–30 are uncovered
`test_specific_bit_positions_on_one` (L325) tests bits 0, 1, 9, 31 only. Other exponent and mantissa bit positions are never verified against known expected values.

### NICE-TO-HAVE: No test for the `None` bit_i random fallback
`bitflip_float32` and `bitflip_float32_fast` accept `bit_i=None` and fall back to `np.random.randint(0, 32)`. Not tested here (or elsewhere in the test suite from what I can see).

---

## Summary

| Severity | Issue |
|---|---|
| CRITICAL | NaN (as input or output) not tested at all |
| CRITICAL | Infinity as input not tested |
| CRITICAL | `inplace=True` code paths not tested |
| IMPORTANT | Denormal/subnormal numbers not tested |
| IMPORTANT | Loose tolerances (L176, L233) mask wrong results |
| IMPORTANT | Cross-impl check only covers sign bit |
| IMPORTANT | No ValueError test for invalid bit positions |
| IMPORTANT | No array exponent/mantissa flip tests |
| NICE-TO-HAVE | Negative zero non-sign-bit flips missing |
| NICE-TO-HAVE | Sloppy tolerance on sign-bit flip (L186) |
| NICE-TO-HAVE | Tolerances inconsistent across file |
| NICE-TO-HAVE | Bits 2–8, 10–30 on 1.0 never checked |
| NICE-TO-HAVE | `bit_i=None` random fallback not tested |
