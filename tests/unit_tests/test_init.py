import importlib
from unittest.mock import patch

import seu_injection.version


def test_dynamic_version_retrieval():
    """Test dynamic version retrieval using importlib.metadata.version."""
    with patch("importlib.metadata.version", return_value="2.0.0"):
        importlib.reload(seu_injection.version)
        from seu_injection.version import __version__

        assert __version__ == "2.0.0", (
            f"Expected dynamic version '2.0.0', got {__version__}"
        )


def test_fallback_version():
    """Test fallback version when dynamic retrieval fails."""
    with patch("importlib.metadata.version", side_effect=Exception):
        importlib.reload(seu_injection.version)
        from seu_injection.version import __version__

        assert __version__ == "1.1.10", (
            f"Expected fallback version '1.1.10', got {__version__}"
        )
