#!/usr/bin/env python3
"""Export platform OpenAPI schema to frontend/openapi/openapi.json."""

import json
from pathlib import Path

from backend.core.app import create_app

OUT = Path(__file__).resolve().parents[2] / "frontend" / "openapi" / "openapi.json"


def main() -> None:
    schema = create_app().openapi()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
