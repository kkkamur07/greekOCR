"""Print kraken vs XML line geometry side by side for one page (sorted by baseline y)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, DOCS, bbox, connect, parse_page_xml

slug, order = sys.argv[1], int(sys.argv[2])
conn, cur = connect()
cur.execute('select id, image_key, width, height from document_parts where document_id=%s and "order"=%s', (DOCS[slug], order))
part = cur.fetchone()
cur.execute('select id, "order", baseline, points, source, kind from lines where part_id=%s order by "order"', (part["id"],))
kl = cur.fetchall(); conn.rollback()
name = os.path.basename(part["image_key"])
_, size, regions = parse_page_xml(os.path.join(DATA, slug, "page", os.path.splitext(name)[0] + ".xml"))
print(f"{slug} page {order} {name} db={part['width']}x{part['height']} xml={size}")
print("--- kraken (order: bbox x0-x1 y0-y1 w h, baseline y mean, npts) ---")
for l in kl:
    x0, x1, y0, y1 = bbox(l["points"]) if l["points"] else (0, 0, 0, 0)
    by = sum(p[1] for p in l["baseline"]["points"]) / len(l["baseline"]["points"]) if l["baseline"]["points"] else -1
    print(f"k{l['order']:<3} x={x0:5.0f}-{x1:5.0f} y={y0:5.0f}-{y1:5.0f} w={x1-x0:5.0f} h={y1-y0:4.0f} by={by:6.0f} bl={len(l['baseline'])} {l['source']}/{l['kind']}")
print("--- xml (region, line: bbox, baseline y mean, text) ---")
for r in regions:
    rb = bbox(r["coords"]) if r["coords"] else None
    print(f"[{r['id']}] region bbox={tuple(round(v) for v in rb) if rb else None} lines={len(r['lines'])} {r['custom']}")
    for l in r["lines"]:
        x0, x1, y0, y1 = bbox(l["coords"]) if l["coords"] else (0, 0, 0, 0)
        by = sum(p[1] for p in l["baseline"]) / len(l["baseline"]) if l["baseline"] else -1
        print(f"  {l['id']:<14} x={x0:5.0f}-{x1:5.0f} y={y0:5.0f}-{y1:5.0f} w={x1-x0:5.0f} by={by:6.0f} | {l['text'][:60]}")
