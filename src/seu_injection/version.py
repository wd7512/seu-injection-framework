"""Version Information Module.

This module defines the version of the SEU Injection Framework, either dynamically retrieved from package metadata or
falling back to a default version for source checkouts.
"""

from importlib.metadata import version as _pkg_version

FALLBACK_VERSION = "1.1.11"  # Fallback version for source checkouts

try:
    __version__ = _pkg_version("seu-injection-framework")
except Exception:
    __version__ = FALLBACK_VERSION
