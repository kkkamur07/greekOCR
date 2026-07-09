#!/usr/bin/env python3
"""Export OpenAPI schema from the FastAPI app to nomicous/frontend/openapi/openapi.json."""

from __future__ import annotations

import json
import sys

from _bootstrap import ensure_nomicous_on_path, nomicous_root


def main() -> int:
    ensure_nomicous_on_path()
    from backend.core.app import create_app

    output = nomicous_root() / "frontend" / "openapi" / "openapi.json"
    schema = create_app().openapi()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
