"""List overlapping segment pairs on a page. usage: overlaps.py <slug> <page_number_1_based> [--render]"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DOCS, SCRATCH, bbox, connect
from shapely.geometry import Polygon

slug, pg = sys.argv[1], int(sys.argv[2]) - 1
conn, cur = connect()
cur.execute('select id, image_key from document_parts where document_id=%s and "order"=%s', (DOCS[slug], pg)); part = cur.fetchone()
cur.execute('''select l.id, l."order", l.points, l.source, l.manual_geometry, l.source_metadata,
    coalesce((select string_agg(lt.text, ' / ') from line_transcriptions lt where lt.line_id=l.id and lt.text<>''), '') as text
    from lines l where part_id=%s order by "order"''', (part["id"],))
lines = cur.fetchall(); conn.rollback()
polys = {str(l["id"]): Polygon(l["points"]).buffer(0) for l in lines}
pairs = []
for i, a in enumerate(lines):
    for b in lines[i+1:]:
        pa, pb = polys[str(a["id"])], polys[str(b["id"])]
        inter = pa.intersection(pb).area
        if inter <= 0: continue
        frac = inter / min(pa.area, pb.area)
        if frac > 0.05: pairs.append((frac, a, b))
pairs.sort(key=lambda t: -t[0])
print(f"{slug} page {pg+1} {os.path.basename(part['image_key'])}: {len(lines)} segments, {len(pairs)} overlapping pairs (>5% of smaller)")
def tag(l):
    m = l["source_metadata"] or {}
    k = "box" if m.get("drawn_by") else ("split" if m.get("split_from") else ("merged" if m.get("merged_from") else l["source"]))
    b = bbox(l["points"]); return f"k{l['order']:<3}[{k:<6}] [{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}] '{l['text'][:22]}'"
for frac, a, b in pairs:
    print(f"  {frac*100:5.1f}%  {tag(a)}  x  {tag(b)}")
if "--render" in sys.argv:
    from PIL import Image, ImageDraw
    im = Image.open(os.path.join(SCRATCH, slug, "images", os.path.basename(part["image_key"]))).convert("RGB")
    sc = 0.4; im = im.resize((int(im.width*sc), int(im.height*sc))); d = ImageDraw.Draw(im)
    bad = {str(l["id"]) for _, a, b in pairs for l in (a, b)}
    for l in lines:
        col = (220, 0, 0) if str(l["id"]) in bad else (0, 150, 60)
        d.polygon([(p[0]*sc, p[1]*sc) for p in l["points"]], outline=col, width=2)
        x0, _, y0, _ = bbox(l["points"]); d.text((x0*sc, y0*sc-10), f"k{l['order']}", fill=col)
    out = os.path.join(SCRATCH, slug, f"overlap_p{pg+1}.png"); im.save(out); print(out)
