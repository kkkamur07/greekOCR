"""Vercel serverless entrypoint for the Nomikos platform API."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "nomikos"))
sys.path.insert(0, str(_ROOT))

from backend.core.app import create_app  # noqa: E402

app = create_app()
