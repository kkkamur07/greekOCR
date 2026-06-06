#!/usr/bin/env python3
"""Export annote OpenAPI schema to frontend/openapi/openapi.json."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from annote.app import create_app

OUT = Path(__file__).resolve().parents[2] / "frontend" / "openapi" / "openapi.json"


def main() -> None:
    with TestClient(create_app()) as client:
        schema = client.get("/openapi.json").json()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
