"""Fix serious segment overlaps (>= 20% of the smaller polygon) on every page of one document.

usage: fix_overlaps.py <slug> [--apply]
Cases: (a) a small segment inside a text line -> cut the small one (dilated 6px) out of the line, keep the
largest remaining piece, clip the baseline to it; (b) two boxes drawn by draw_boxes.py stacked over each
other -> trim both at the midpoint of the overlap; (c) a drawn box duplicating a hand-drawn line with the
same text -> delete the box and pair the text line to the hand-drawn one. Hand-drawn lines are never edited.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psycopg2.extras
from common import DOCS, bbox, connect
from shapely.geometry import Polygon
from shapely.geometry import box as sbox

slug = sys.argv[1]; APPLY = "--apply" in sys.argv; THR = 0.20
conn, cur = connect()
cur.execute("select id from transcriptions where document_id=%s and kind='ground_truth'", (DOCS[slug],)); gt = cur.fetchone()["id"]
cur.execute('select id, "order", image_key from document_parts where document_id=%s order by "order"', (DOCS[slug],))
parts = cur.fetchall()

def clip_baseline(bl, x0, x1):
    pts = sorted([list(p) for p in bl])
    if len(pts) < 2: return pts
    def y_at(x):
        for (ax, ay), (bx, by) in zip(pts, pts[1:], strict=False):
            if ax <= x <= bx: return ay if bx == ax else ay + (by - ay) * (x - ax) / (bx - ax)
        return pts[0][1] if x < pts[0][0] else pts[-1][1]
    lo, hi = max(x0, pts[0][0]), min(x1, pts[-1][0])
    if hi - lo < 10: return [[x0, y_at(x0)], [x1, y_at(x1)]]
    return [[round(lo, 1), round(y_at(lo), 1)]] + [p for p in pts if lo < p[0] < hi] + [[round(hi, 1), round(y_at(hi), 1)]]

def kind(l):
    m = l["source_metadata"] or {}
    return "box" if m.get("drawn_by") else ("hand" if l["source"] == "manual" else "auto")

def save_poly(l, poly):
    pts = [[round(x, 1), round(y, 1)] for x, y in poly.exterior.coords[:-1]]
    x0, x1, y0, y1 = bbox(pts)
    bl = clip_baseline((l["baseline"] or {}).get("points", []), x0, x1)
    meta = dict(l["source_metadata"] or {}); meta["overlap_fixed_by"] = "scripts/armenian/fix_overlaps.py"
    cur.execute("update lines set points=%s, mask=%s, baseline=%s, kraken_ceiling=%s, source_metadata=%s where id=%s",
                (psycopg2.extras.Json(pts), psycopg2.extras.Json({"points": pts}), psycopg2.extras.Json({"points": bl}),
                 psycopg2.extras.Json(pts + [pts[0]]) if l["kraken_ceiling"] else None, psycopg2.extras.Json(meta), l["id"]))

n_fix = 0
for part in parts:
    cur.execute('''select l.*, coalesce((select string_agg(lt.text, ' / ') from line_transcriptions lt where lt.line_id=l.id and lt.text<>''), '') as text
                   from lines l where part_id=%s order by "order"''', (part["id"],))
    lines = cur.fetchall()
    polys = {str(l["id"]): Polygon(l["points"]).buffer(0) for l in lines}
    for i, a in enumerate(lines):
        for b in lines[i+1:]:
            pa, pb = polys[str(a["id"])], polys[str(b["id"])]
            if pa.is_empty or pb.is_empty: continue
            inter = pa.intersection(pb).area
            if inter <= 0 or inter / min(pa.area, pb.area) < THR: continue
            small, big = (a, b) if pa.area <= pb.area else (b, a)
            ps, pg_ = polys[str(small["id"])], polys[str(big["id"])]
            frac = inter / ps.area
            ka, kb = kind(small), kind(big)
            tag = f"page {part['order']+1:>2} k{small['order']}[{ka}] '{small['text'][:14]}' in k{big['order']}[{kb}] '{big['text'][:16]}' {frac*100:.0f}%"
            if ka == "box" and kb == "hand" and small["text"].strip() == big["text"].strip() and small["text"]:
                print(f"{tag}: DUPLICATE -> delete box, pair text to hand-drawn line"); n_fix += 1
                if APPLY:
                    cur.execute("update page_transcription_lines set paired_line_id=%s where paired_line_id=%s", (big["id"], small["id"]))
                    cur.execute("insert into line_transcriptions (id, line_id, transcription_id, text, confidence) values (gen_random_uuid(), %s, %s, %s, null) on conflict (line_id, transcription_id) do update set text=excluded.text", (big["id"], gt, small["text"]))
                    cur.execute("delete from lines where id=%s", (small["id"],)); polys[str(small["id"])] = Polygon()
                continue
            if ka == "box" and kb == "box":
                sx0, sx1, sy0, sy1 = bbox(small["points"]); bx0, bx1, by0, by1 = bbox(big["points"])
                upper, lower = (small, big) if sy0 <= by0 else (big, small)
                ux0, ux1, uy0, uy1 = bbox(upper["points"]); lx0, lx1, ly0, ly1 = bbox(lower["points"])
                mid = (ly0 + uy1) / 2
                print(f"{tag}: STACKED BOXES -> trim at y={mid:.0f}"); n_fix += 1
                if APPLY:
                    save_poly(upper, sbox(ux0, uy0, ux1, mid - 2)); save_poly(lower, sbox(lx0, mid + 2, lx1, ly1))
                    polys[str(upper["id"])] = sbox(ux0, uy0, ux1, mid - 2); polys[str(lower["id"])] = sbox(lx0, mid + 2, lx1, ly1)
                continue
            if kb == "hand":
                print(f"{tag}: both/large hand-drawn, LEFT ALONE"); continue
            sx0, sx1, sy0, sy1 = bbox(small["points"]); gx0, gx1, gy0, gy1 = bbox(big["points"])
            if (sx1 - sx0) > 0.5 * (gx1 - gx0) and (sy1 - sy0) > 0.5 * (gy1 - gy0):
                print(f"{tag}: two text lines touching, LEFT ALONE"); continue
            M = 6; huge = 1e6
            cands = [pg_.intersection(sbox(-huge, -huge, sx0 - M, huge)), pg_.intersection(sbox(sx1 + M, -huge, huge, huge)),
                     pg_.intersection(sbox(-huge, -huge, huge, sy0 - M)), pg_.intersection(sbox(-huge, sy1 + M, huge, huge))]
            cands = [max(c.geoms, key=lambda g: g.area) if c.geom_type != "Polygon" else c for c in cands if not c.is_empty]
            cut = max(cands, key=lambda c: c.area).simplify(1.0)
            if cut.area < 0.6 * pg_.area:
                print(f"{tag}: clipping would keep only {cut.area/pg_.area*100:.0f}%, LEFT ALONE"); continue
            b1 = [round(v) for v in cut.bounds]
            print(f"{tag}: CLIP line away from numeral: [{gx0:.0f},{gx1:.0f},{gy0:.0f},{gy1:.0f}] -> [{b1[0]},{b1[2]},{b1[1]},{b1[3]}] ({cut.area/pg_.area*100:.0f}% area kept)"); n_fix += 1
            if APPLY:
                save_poly(big, cut); polys[str(big["id"])] = cut
print(f"{n_fix} fixes"); 
if APPLY: conn.commit(); print("COMMITTED")
else: conn.rollback(); print("DRY RUN")
