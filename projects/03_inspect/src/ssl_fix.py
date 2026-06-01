"""ssl_fix.py — Must be imported before inspect_ai or httpx on Windows."""
import os
from pathlib import Path


def apply() -> None:
    cert_env = os.environ.get("SSL_CERT_FILE", "")
    if cert_env and not Path(cert_env).exists():
        try:
            import certifi
            os.environ["SSL_CERT_FILE"] = certifi.where()
        except ImportError:
            pass
