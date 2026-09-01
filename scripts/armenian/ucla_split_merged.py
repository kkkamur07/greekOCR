"""Split kraken segments that merged several XML text lines side by side (MS_UCLA_MS).

usage: ucla_split_merged.py [--pages 1,3] [--apply]
Reads the current plan JSON for each page. For every kraken line decided 'noise' whose polygon
covers two or more XML text lines lying side by side (vertical offset < thr, >= 60 % of the XML
extent inside the polygon), the polygon and baseline are clipped at the midpoints between the
covered lines. The first piece overwrites the original row, the others are inserted as new
kraken rows on the same part/block. Re-run plan.py afterwards so the pieces get paired.
Dry run by default; --apply commits. Never touches manual_geometry lines.
"""
import glob
import os
import statistics
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psycopg2.extras
from common import DATA, SCRATCH, bbox, connect, load_json, parse_page_xml

SLUG = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "ms_ucla"
APPLY = "--apply" in sys.argv
only = {int(v) for v in sys.argv[sys.argv.index("--pages") + 1].split(",")} if "--pages" in sys.argv else None


def y_at(poly, x):
    poly = sorted(poly)
    if x <= poly[0][0]:
        return poly[0][1]
    if x >= poly[-1][0]:
        return poly[-1][1]
    for (ax, ay), (bx, by) in zip(poly, poly[1:], strict=False):
        if ax <= x <= bx:
            return ay if bx == ax else ay + (by - ay) * (x - ax) / (bx - ax)
    return poly[-1][1]


def mean_dy(kg, xg, lo, hi):
    xs = [lo + (hi - lo) * i / 10 for i in range(11)]
    return statistics.mean(abs(y_at(kg, x) - y_at(xg, x)) for x in xs)


def clip_half(poly, x_cut, keep_left):
    """Sutherland-Hodgman against the vertical line x = x_cut."""
    inside = (lambda p: p[0] <= x_cut) if keep_left else (lambda p: p[0] >= x_cut)
    out = []
    for i in range(len(poly)):
        cur, prev = poly[i], poly[i - 1]
        if inside(cur):
            if not inside(prev):
                out.append(intersect(prev, cur, x_cut))
            out.append(cur)
        elif inside(prev):
            out.append(intersect(prev, cur, x_cut))
    return out


def intersect(a, b, x_cut):
    (ax, ay), (bx, by) = a, b
    t = (x_cut - ax) / (bx - ax) if bx != ax else 0.0
    return [round(x_cut, 1), round(ay + t * (by - ay), 1)]


def clip_poly(poly, lo, hi):
    p = [list(q) for q in poly]
    if lo is not None:
        p = clip_half(p, lo, keep_left=False)
    if hi is not None and p:
        p = clip_half(p, hi, keep_left=True)
    return p


def clip_baseline(bl, lo, hi):
    pts = sorted([list(q) for q in bl])
    lo = pts[0][0] if lo is None else max(lo, pts[0][0])
    hi = pts[-1][0] if hi is None else min(hi, pts[-1][0])
    if hi - lo < 10:
        return []
    inner = [q for q in pts if lo < q[0] < hi]
    return [[round(lo, 1), round(y_at(pts, lo), 1)]] + inner + [[round(hi, 1), round(y_at(pts, hi), 1)]]


conn, cur = connect()
todo = []
for path in sorted(glob.glob(os.path.join(SCRATCH, SLUG, "plan", "*.json"))):
    plan = load_json(path)
    if only is not None and plan["order"] not in only:
        continue
    _, _, regions = parse_page_xml(os.path.join(DATA, SLUG, "page", os.path.splitext(plan["image"])[0] + ".xml"))
    xlines = []
    for r in regions:
        for l in r["lines"]:
            geo = l["baseline"] if len(l["baseline"]) >= 2 else l["coords"]
            if geo and l["text"]:
                b = bbox(geo)
                if (b[1] - b[0]) >= (b[3] - b[2]):
                    xlines.append((l["id"], [tuple(q) for q in geo], b, l["text"]))
    cur.execute('select * from lines where part_id=%s', (plan["part_id"],))
    for k in cur.fetchall():
        d = plan["kraken"].get(str(k["id"]))
        if not d or k["manual_geometry"]:
            continue
        if d["decision"] != "noise" and d.get("reason") != "override":
            continue
        kg = [tuple(q) for q in (k["baseline"] or {}).get("points", [])]
        if len(kg) < 2 or not k["points"]:
            continue
        kb = bbox(k["points"])
        covered = []
        for xid, xg, xb, text in xlines:
            lo, hi = max(kb[0], xb[0]), min(kb[1], xb[1])
            if hi <= lo or (hi - lo) / max(1, xb[1] - xb[0]) < 0.6:
                continue
            if xb[2] < kb[2] - 40 or xb[3] > kb[3] + 40:
                continue
            if mean_dy(kg, xg, lo, hi) < plan["thr"]:
                covered.append((xb[0], xb[1], xid, text))
        covered.sort()
        if len(covered) < 2:
            continue
        ok = all(a[1] < b[0] for a, b in zip(covered, covered[1:], strict=False))
        cuts = [(a[1] + b[0]) / 2 for a, b in zip(covered, covered[1:], strict=False)] if ok else []
        todo.append((plan["order"], plan["image"], k, covered, cuts, ok))

print(f"{len(todo)} merged segments")
for order, image, k, covered, cuts, ok in todo:
    kb = [round(v) for v in bbox(k["points"])]
    print(f"page {order:>2} {image} k{k['order']:<3} {str(k['id'])[:8]} bbox={kb} -> {len(covered)} pieces, cuts={[round(c) for c in cuts]}" + ("" if ok else "  OVERLAPPING EXTENTS, SKIPPED"))
    for x0, x1, xid, text in covered:
        print(f"      {xid:<12} x={x0:.0f}-{x1:.0f} | {text}")
    if not ok:
        continue
    bounds = [None] + cuts + [None]
    pieces = []
    for i in range(len(covered)):
        lo, hi = bounds[i], bounds[i + 1]
        poly = clip_poly(k["points"], lo, hi)
        bl = clip_baseline((k["baseline"] or {}).get("points", []), lo, hi)
        if len(poly) < 3 or len(bl) < 2:
            print(f"      piece {i}: degenerate (poly={len(poly)} bl={len(bl)}), SKIPPING WHOLE SEGMENT")
            pieces = None
            break
        pieces.append((poly, bl))
    if pieces is None or not APPLY:
        continue
    meta = dict(k["source_metadata"] or {})
    for i, (poly, bl) in enumerate(pieces):
        m = {**meta, "split_from": str(k["id"]), "split_index": i, "split_by": "scripts/armenian/ucla_split_merged.py"}
        ring = [list(q) for q in poly] + [list(poly[0])]
        if i == 0:
            cur.execute("update lines set points=%s, mask=%s, baseline=%s, kraken_ceiling=%s, source_metadata=%s where id=%s",
                        (psycopg2.extras.Json(poly), psycopg2.extras.Json({"points": poly}), psycopg2.extras.Json({"points": bl}),
                         psycopg2.extras.Json(ring), psycopg2.extras.Json(m), k["id"]))
        else:
            cur.execute("insert into lines (id, part_id, block_id, baseline, mask, kind, points, source, source_metadata, kraken_ceiling, manual_geometry, \"order\") "
                        "values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, false, %s)",
                        (uuid.uuid4(), k["part_id"], k["block_id"], psycopg2.extras.Json({"points": bl}), psycopg2.extras.Json({"points": poly}),
                         k["kind"], psycopg2.extras.Json(poly), k["source"], psycopg2.extras.Json(m), psycopg2.extras.Json(ring), k["order"]))
if APPLY:
    conn.commit(); print("COMMITTED")
else:
    conn.rollback(); print("DRY RUN - nothing written")
