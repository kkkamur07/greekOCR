"""Merge kraken split fragments into the segment that carries the text (both Armenian documents).

usage: merge_splits.py <slug> [--pages 0,3] [--apply]
Reads plan.py output: every kraken line with decision 'split' is merged into the 'pair' line of
the same XML text line. The merged polygon is the union of both masks with the gap between them
bridged; the baselines are joined; the primary row keeps its id (so its pairing and ground truth
survive) and the fragment row is deleted. Dry run renders before/after crops to SCRATCH/<slug>/merge/.
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psycopg2.extras
from common import SCRATCH, bbox, connect, load_json
from shapely.geometry import LineString, Polygon
from shapely.ops import nearest_points, unary_union

slug = sys.argv[1]
APPLY = "--apply" in sys.argv
only = {int(v) for v in sys.argv[sys.argv.index("--pages") + 1].split(",")} if "--pages" in sys.argv else None


def merged_polygon(a, b):
    """Union of both masks plus a band bridging the gap between them (the gap is the line's own text)."""
    pa, pb = Polygon(a).buffer(0), Polygon(b).buffer(0)
    parts = [pa, pb]
    if pa.distance(pb) > 0:
        p1, p2 = nearest_points(pa, pb)
        ha, hb = bbox(a)[3] - bbox(a)[2], bbox(b)[3] - bbox(b)[2]
        parts.append(LineString([p1, p2]).buffer(0.45 * min(ha, hb), cap_style="flat"))
    u = unary_union(parts).buffer(3, join_style="mitre").buffer(-3, join_style="mitre")
    if u.geom_type != "Polygon":
        u = max(u.geoms, key=lambda g: g.area)
    u = u.simplify(1.5)
    return [[round(x, 1), round(y, 1)] for x, y in u.exterior.coords[:-1]]


def merged_baseline(a, b, vertical):
    pts = [tuple(p) for p in a] + [tuple(p) for p in b]
    axis = 1 if vertical else 0
    pts.sort(key=lambda p: p[axis])
    out = [pts[0]]
    for p in pts[1:]:
        if abs(p[axis] - out[-1][axis]) >= 4:
            out.append(p)
    line = LineString(out).simplify(1.0) if len(out) > 2 else LineString(out)
    return [[round(x, 1), round(y, 1)] for x, y in line.coords]


conn, cur = connect()
jobs = []
for path in sorted(glob.glob(os.path.join(SCRATCH, slug, "plan", "*.json"))):
    plan = load_json(path)
    if only is not None and plan["order"] not in only:
        continue
    prim_of = {d["x"]: kid for kid, d in plan["kraken"].items() if d["decision"] == "pair"}
    for kid, d in plan["kraken"].items():
        if d["decision"] != "split":
            continue
        p = prim_of.get(d["x"])
        if not p:
            print(f"page {plan['order']}: fragment {kid[:8]} has no primary for {d['x']}, skipped"); continue
        jobs.append((plan, kid, p, d["x"]))
print(f"{slug}: {len(jobs)} fragments to merge")
if not APPLY:
    from PIL import Image, ImageDraw
    os.makedirs(os.path.join(SCRATCH, slug, "merge"), exist_ok=True)
for plan, frag_id, prim_id, xid in jobs:
    cur.execute("select id, points, baseline, kraken_ceiling, source_metadata, manual_geometry, source, "
                "(select count(*) from line_transcriptions lt where lt.line_id = lines.id and lt.text <> '') as gt, "
                "(select count(*) from page_transcription_lines t where t.paired_line_id = lines.id) as paired "
                "from lines where id in (%s, %s)", (frag_id, prim_id))
    rows = {str(r["id"]): r for r in cur.fetchall()}
    if len(rows) != 2:
        print(f"page {plan['order']}: rows missing for {frag_id[:8]}/{prim_id[:8]} (already merged?), skipped"); continue
    frag, prim = rows[frag_id], rows[prim_id]
    if prim["manual_geometry"] or frag["manual_geometry"] or prim["source"] != "kraken" or frag["source"] != "kraken":
        print(f"page {plan['order']}: {prim_id[:8]}/{frag_id[:8]} edited by hand or not kraken, skipped"); continue
    if frag["gt"] or frag["paired"]:
        print(f"page {plan['order']}: fragment {frag_id[:8]} has its own pairing/text, skipped"); continue
    poly = merged_polygon(prim["points"], frag["points"])
    pb, fb = prim["baseline"]["points"], frag["baseline"]["points"]
    x0, x1, y0, y1 = bbox(poly)
    bl = merged_baseline(pb, fb, vertical=(y1 - y0) > (x1 - x0))
    text = next(t["text"] for t in plan["text_lines"] if t["xml_id"] == xid)
    print(f"page {plan['order']:>2} {plan['image']:<14} frag {[round(v) for v in bbox(frag['points'])]} + prim {[round(v) for v in bbox(prim['points'])]} -> {[round(v) for v in (x0, x1, y0, y1)]} pts={len(poly)} bl={len(bl)} | {text}")
    if APPLY:
        meta = dict(prim["source_metadata"] or {})
        meta.setdefault("merged_from", []).append(frag_id)
        meta["merged_by"] = "scripts/armenian/merge_splits.py"
        cur.execute("update lines set points=%s, mask=%s, baseline=%s, kraken_ceiling=%s, source_metadata=%s where id=%s",
                    (psycopg2.extras.Json(poly), psycopg2.extras.Json({"points": poly}), psycopg2.extras.Json({"points": bl}),
                     psycopg2.extras.Json(poly + [poly[0]]), psycopg2.extras.Json(meta), prim_id))
        cur.execute("delete from lines where id=%s", (frag_id,))
    else:
        im = Image.open(os.path.join(SCRATCH, slug, "images", plan["image"])).convert("RGB")
        pad = 60
        X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(im.width, x1 + pad), min(im.height, y1 + pad)
        c = im.crop((X0, Y0, X1, Y1)); w = c.width
        canvas = Image.new("RGB", (w * 2 + 10, c.height), (255, 255, 255)); canvas.paste(c, (0, 0)); canvas.paste(c, (w + 10, 0))
        d = ImageDraw.Draw(canvas)
        sh = lambda pts, dx, X0=X0, Y0=Y0: [(p[0] - X0 + dx, p[1] - Y0) for p in pts]
        d.polygon(sh(prim["points"], 0), outline=(0, 150, 60), width=3); d.polygon(sh(frag["points"], 0), outline=(235, 120, 0), width=3)
        d.line(sh(pb, 0), fill=(0, 150, 60), width=3); d.line(sh(fb, 0), fill=(235, 120, 0), width=3)
        d.polygon(sh(poly, w + 10), outline=(0, 90, 220), width=3); d.line(sh(bl, w + 10), fill=(0, 90, 220), width=3)
        canvas.save(os.path.join(SCRATCH, slug, "merge", f"{plan['order']:02d}_{frag_id[:8]}.png"))
if APPLY:
    conn.commit(); print("COMMITTED")
else:
    conn.rollback(); print("DRY RUN - nothing written; crops in", os.path.join(SCRATCH, slug, "merge"))
