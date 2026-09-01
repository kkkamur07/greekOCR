"""Shared helpers for the Armenian (MS_P_172, MS_UCLA_MS) segment cleanup and pairing.

Loads nomikos/backend/core/.env.supabase, opens a psycopg2 connection on the migrator URL,
and can pull page images from Supabase storage. Read-only unless a caller commits.
"""
import json
import os
import sys
from xml.etree import ElementTree as ET

import psycopg2
import psycopg2.extras
from psycopg2.extras import register_uuid

register_uuid()

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV_FILE = os.environ.get("SUPABASE_ENV_FILE", os.path.join(ROOT, "nomikos/backend/core/.env.supabase"))
DATA = os.path.join(ROOT, "data/dataset/armenian")
SCRATCH = os.environ.get(
    "ARMENIAN_SCRATCH",
    "/private/tmp/claude-501/-Users-krishuagarwal-Desktop-Programming-python-greekOCR/8f1472c5-31ae-475e-9d90-e99eed1235d1/scratchpad/armenian",
)

DOCS = {
    "ms_p_172": "7e718475-2c3e-4fc4-9e06-63f39786551a",   # MS_P_172
    "ms_ucla": "cd87776c-fc8e-4005-9799-675611f65a38",    # MS_UCLA_MS
}
PAGE_NS = "{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}"


def load_env():
    with open(ENV_FILE) as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key, value)


def connect():
    load_env()
    conn = psycopg2.connect(os.environ["MIGRATOR_DATABASE_URL"])
    return conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)


def media_store():
    load_env()
    sys.path.insert(0, os.path.join(ROOT, "nomikos"))
    from backend.document.infrastructure.media_store.supabase import (
        SupabaseMediaStore,  # noqa: E402
    )
    return SupabaseMediaStore()


def fetch_image(store, image_key, dest):
    if os.path.exists(dest):
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as handle:
        handle.write(store.read(image_key))
    return dest


def parse_points(text):
    return [[float(a), float(b)] for a, b in (pair.split(",") for pair in text.split())]


def parse_page_xml(path):
    """Return (image_filename, (w, h), regions) where a region is {id, coords, lines:[{id, coords, baseline, text, index}]}."""
    root = ET.parse(path).getroot()  # noqa: S314 - our own exported PAGE XML from the local dataset, not untrusted input
    page = root.find(PAGE_NS + "Page")
    size = (int(page.get("imageWidth")), int(page.get("imageHeight")))
    regions = []
    for region in page.findall(PAGE_NS + "TextRegion"):
        coords = region.find(PAGE_NS + "Coords")
        lines = []
        for line in region.findall(PAGE_NS + "TextLine"):
            lc = line.find(PAGE_NS + "Coords")
            bl = line.find(PAGE_NS + "Baseline")
            unicode_el = line.find(PAGE_NS + "TextEquiv/" + PAGE_NS + "Unicode")
            text = (unicode_el.text or "") if unicode_el is not None else ""
            lines.append({
                "id": line.get("id"),
                "coords": parse_points(lc.get("points")) if lc is not None and lc.get("points") else [],
                "baseline": parse_points(bl.get("points")) if bl is not None and bl.get("points") else [],
                "text": text.strip(),
                "custom": line.get("custom", ""),
            })
        regions.append({"id": region.get("id"), "coords": parse_points(coords.get("points")) if coords is not None else [], "lines": lines, "custom": region.get("custom", "")})
    return page.get("imageFilename"), size, regions


def bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), max(xs), min(ys), max(ys)


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def dump_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=1, default=str)
