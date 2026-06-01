# conftest.py — Applies SSL cert fix before any test imports langchain_ollama.
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
