You are an API design & compatibility reviewer reviewing a PR that adds hardcoded unit tests.

Focus on:
- Are all three implementations (bitflip_float32, bitflip_float32_optimized, bitflip_float32_fast) tested consistently?
- Is the bit indexing convention clearly documented?
- Do the tests match the existing test patterns/style?

Provide findings as a structured list with severity (CRITICAL/IMPORTANT/NICE-TO-HAVE).

The diff is at tests/unit_tests/test_bitflip_hardcoded.py — read it directly.
