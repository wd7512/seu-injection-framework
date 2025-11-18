import importlib
from unittest.mock import patch

import seu_injection  # still import main package if needed


def test_dynamic_version_retrieval():
    with patch("importlib.metadata.version", return_value="2.0.0"):
        import seu_injection.version  # import inside patch context

        importlib.reload(seu_injection.version)
        from seu_injection.version import __version__

        assert __version__ == "2.0.0", f"Expected dynamic version '2.0.0', got {__version__}"


def test_fallback_version():
    with patch("importlib.metadata.version", side_effect=Exception):
        import seu_injection.version

        importlib.reload(seu_injection.version)
        from seu_injection.version import __version__

        assert __version__ == "1.1.10", f"Expected fallback version '1.1.10', got {__version__}"
