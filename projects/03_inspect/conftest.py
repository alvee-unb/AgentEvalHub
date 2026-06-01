# conftest.py — SSL fix before any inspect_ai / httpx import
import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

cert_env = os.environ.get("SSL_CERT_FILE", "")
if cert_env and not Path(cert_env).exists():
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
    except ImportError:
        pass
