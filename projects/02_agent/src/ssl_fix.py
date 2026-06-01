"""
ssl_fix.py — Patches SSL_CERT_FILE to use certifi on Windows when the
system-configured path no longer exists.  Import this before any httpx/ollama
import.
"""

import os
from pathlib import Path


def apply() -> None:
    cert_env = os.environ.get("SSL_CERT_FILE", "")
    if cert_env and not Path(cert_env).exists():
        try:
            import certifi
            os.environ["SSL_CERT_FILE"] = certifi.where()
        except ImportError:
            pass  # certifi not installed — let it fail naturally
