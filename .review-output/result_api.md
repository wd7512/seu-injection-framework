[0m
> build · big-pickle
[0m
[0m→ [0mRead tests/unit_tests/test_bitflip_hardcoded.py
[0m✱ [0mGlob "tests/**/*.py"[90m in . · 27 matches[0m
[0m→ [0mRead tests/unit_tests/test_bitflip.py
[0m→ [0mRead tests/unit_tests/test_bitflip_optimized.py
[0m→ [0mRead tests/unit_tests/test_injector.py
[0m✗ [0mRead src/seu_injection/bitops.py failed
[91m[1mError: [0mFile not found: /Users/williamdennis/repos/seu-injection-framework/src/seu_injection/bitops.py

Did you mean one of these?
/Users/williamdennis/repos/seu-injection-framework/src/seu_injection/bitops
[0m→ [0mRead tests/unit_tests/conftest.py
[0m→ [0mRead tests/unit_tests/test_utils.py
Let me check the actual bitops module to compare indexing conventions with the implementation.
[0m→ [0mRead src/seu_injection/bitops
[0m→ [0mRead src/seu_injection/bitops/float32.py
[0m→ [0mRead src/seu_injection/bitops/__init__.py
[0m→ [0mRead src/seu_injection/bitops/float32_legacy.py
Now I have a thorough view of the full codebase. Here is my review report:

---

## API Design & Compatibility Review: `test_bitflip_hardcoded.py`

### CRITICAL

**1. Only 4 of 19 tests exercise all 3 implementations.**
Lines 31-48, 53-65, 106-112, 235-258 are the sole tests using all three functions (`bitflip_float32`, `bitflip_float32_optimized`, `bitflip_float32_fast`). The remaining 15 tests only call `bitflip_float32` (the legacy/string-based path), defeating the purpose of a hardcoded regression suite. For example:
- `test_exponent_bit_flip_known_value` (line 114) — only legacy
- `test_mantissa_lsb_flip_known_value` (line 132) — only legacy
- `test_zero_sign_bit_flip` (line 149) — only legacy
- `test_specific_small_value_flip` (line 164) — only legacy
- `test_large_value_sign_flip` (line 178) — only legacy
- `test_sequence_of_specific_flips` (line 218) — only legacy
- `test_specific_bit_positions_on_one` (line 325) — only legacy

**2. `test_power_of_two_values` (line 298) and `test_fractional_powers_of_two` (line 312) each test only 2 of 3 implementations, and the pairs are *different*.**
`test_power_of_two_values` tests `bitflip_float32` + `bitflip_float32_optimized` but omits `_fast`. `test_fractional_powers_of_two` tests `bitflip_float32` + `bitflip_float32_fast` but omits `_optimized`. This asymmetric coverage looks like an oversight and leaves real gaps: if `bitflip_float32_fast` regresses on powers-of-two, no hardcoded test catches it.

**3. `test_all_implementations_consistent_hardcoded` (line 235) only tests bit 0 (sign bit).**
Despite its name, it only covers sign-bit flips on 5 scalar values. It does not test exponent or mantissa bit positions cross-implementation. A test named "all implementations consistent" should sample across the bit range (e.g., bits 0, 1, 8, 15, 23, 31).

### IMPORTANT

**4. `test_matrix_specific_position_flip` (line 67) hardcodes a manual 2-step extraction + assignment pattern rather than testing a vectorized call.**
The test extracts a single element, flips it, then assigns it back. This tests user code, not the API contract. Consider also adding a direct assertion like `np.testing.assert_array_equal(bitflip_float32(matrix, 0), expected)` which is what real callers actually do.

**5. Duplicates coverage from `test_bitflip.py`.**
`test_bitflip.py:test_bitflip_float32_basic` tests `bitflip_float32(1.0, 0) == -1.0`, which is the same first assertion in `test_sign_bit_flip_positive_to_negative`. Similar overlap exists for array sign flips. While duplication isn't harmful if intentional, there is no documentation explaining what unique value this file provides over `test_bitflip.py`. The module docstring says "strong guarantees about bitflip correctness by using predetermined values," but `test_bitflip.py` already uses deterministic assertions.

**6. No error-path testing for any implementation.**
`test_bitflip.py` and `test_bitflip_optimized.py` both test invalid bit positions with `pytest.raises(ValueError, ...)`. This file tests zero error paths. Since the file is described as "hardcoded" tests, including hardcoded error cases (e.g., `bitflip_float32(1.0, -1) → ValueError`) is worth considering.

### NICE-TO-HAVE

**7. Single-line import vs. multi-line import in sister files.**
This file uses a single-line import:
```python
from seu_injection.bitops import bitflip_float32, bitflip_float32_fast, bitflip_float32_optimized
```
while `test_bitflip.py` uses multi-line parenthesized imports for the same module. Minor, but inconsistent.

**8. `test_large_value_sign_flip` (line 178) uses a loose tolerance (`abs(result + 123456.78) < 0.01`).**
For a "hardcoded" test file, the expectation should be exact equality. The value 123456.78 is not exactly representable in float32 (it rounds to 123456.78125), so the assertion will always have error. Either the expected value should be replaced with the true float32 representation, or the test should explicitly note the rounding and use `np.testing.assert_allclose` with `atol` set accordingly.

**9. `test_sequence_of_specific_flips` (line 218) is the only stateful/stepwise test.**
It tests a 3-step sequence on the same value. This pattern is good for regression but is not replicated for the `_optimized` or `_fast` paths. Consider adding a parametrized version.
