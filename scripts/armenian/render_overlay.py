"""Render kraken (red) and XML (blue) baselines on a page image for visual comparison.

usage: render_overlay.py <slug> <page_order> [--lines-json path]  -> writes SCRATCH/<slug>/overlay/<order>.png
If --lines-json is given, kraken lines are taken from that file (list of {id, baseline, points, order}) instead of the DB.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, DOCS, SCRATCH, connect, load_json, parse_page_xml
from PIL import Image, ImageDraw, ImageFont

slug, order = sys.argv[1], int(sys.argv[2])
scale = 0.35
conn, cur = connect()
cur.execute('select id, image_key from document_parts where document_id=%s and "order"=%s', (DOCS[slug], order))
part = cur.fetchone()
name = os.path.basename(part["image_key"])
if "--lines-json" in sys.argv:
    klines = load_json(sys.argv[sys.argv.index("--lines-json") + 1])
else:
    cur.execute('select id, "order", baseline, points from lines where part_id=%s order by "order"', (part["id"],))
    klines = cur.fetchall()
conn.rollback()
xml_path = os.path.join(DATA, slug, "page", os.path.splitext(name)[0] + ".xml")
_, size, regions = parse_page_xml(xml_path)
img = Image.open(os.path.join(SCRATCH, slug, "images", name)).convert("RGB")
img = img.resize((int(img.width * scale), int(img.height * scale)))
draw = ImageDraw.Draw(img)
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
except Exception:
    font = ImageFont.load_default()
def sc(pts): return [(p[0] * scale, p[1] * scale) for p in pts]
for _i, l in enumerate(klines):
    if l.get("points"):
        draw.polygon(sc(l["points"]), outline=(255, 0, 0))
    if l.get("baseline") and len(l["baseline"]["points"]) > 1:
        draw.line(sc(l["baseline"]["points"]), fill=(255, 0, 0), width=2)
        x, y = sc(l["baseline"]["points"])[0]
        draw.text((x - 22, y - 8), f"k{l['order']}", fill=(200, 0, 0), font=font)
n = 0
for r in regions:
    if r["coords"]:
        draw.polygon(sc(r["coords"]), outline=(0, 160, 0))
    for l in r["lines"]:
        if l["baseline"] and len(l["baseline"]) > 1:
            draw.line(sc(l["baseline"]), fill=(0, 0, 255), width=2)
            x, y = sc(l["baseline"])[-1]
            draw.text((x + 4, y - 8), f"x{n}", fill=(0, 0, 220), font=font)
        n += 1
out = os.path.join(SCRATCH, slug, "overlay", f"{order:02d}_{os.path.splitext(name)[0]}.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
img.save(out)
print(out)
