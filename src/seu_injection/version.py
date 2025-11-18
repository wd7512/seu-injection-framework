from importlib.metadata import version as _pkg_version

FALLBACK_VERSION = "1.1.10"  # Fallback version for source checkouts

try:
    __version__ = _pkg_version("seu-injection-framework")
except Exception:
    __version__ = FALLBACK_VERSION
