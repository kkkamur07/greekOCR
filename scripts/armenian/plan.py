"""Match kraken segments (DB) to Transkribus XML lines (text) for one Armenian document.

usage: plan.py <slug> [--pages 0,3,5] [--render]
Writes SCRATCH/<slug>/plan/<order>.json and prints a per-page summary. Read-only.

Per kraken line K the decision is one of:
  pair      K sits on exactly one XML line X and is X's primary piece -> gets X's text
  split     K sits on X but another K covers X better (kraken split the line) -> kept, unpaired, flagged
  noise     K sits on no XML line -> delete
  keep      K sits on an XML line whose text is empty -> kept, unpaired
Overrides (SCRATCH/<slug>/overrides.json) can force per line id: {"<line_id>": "noise"|"keep"|"pair:<xml_line_id>"}.
"""
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, DOCS, SCRATCH, bbox, connect, dump_json, load_json, parse_page_xml

slug = sys.argv[1]
only = None
if "--pages" in sys.argv:
    only = {int(v) for v in sys.argv[sys.argv.index("--pages") + 1].split(",")}
RENDER = "--render" in sys.argv
ov_path = os.path.join(SCRATCH, slug, "overrides.json")
overrides = load_json(ov_path) if os.path.exists(ov_path) else {}


def seg_dist(p, a, b):
    (px, py), (ax, ay), (bx, by) = p, a, b
    dx, dy = bx - ax, by - ay
    if dx == dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def poly_dist(p, poly):
    if len(poly) == 1:
        return math.hypot(p[0] - poly[0][0], p[1] - poly[0][1])
    return min(seg_dist(p, poly[i], poly[i + 1]) for i in range(len(poly) - 1))


def sample(poly, step=8.0):
    if len(poly) < 2:
        return list(poly)
    out = []
    for i in range(len(poly) - 1):
        (ax, ay), (bx, by) = poly[i], poly[i + 1]
        n = max(1, int(math.hypot(bx - ax, by - ay) / step))
        for k in range(n):
            t = k / n
            out.append((ax + t * (bx - ax), ay + t * (by - ay)))
    out.append(tuple(poly[-1]))
    return out


def is_vertical(poly):
    x0, x1, y0, y1 = bbox(poly)
    return (y1 - y0) > (x1 - x0)


def extent(poly, vertical):
    x0, x1, y0, y1 = bbox(poly)
    return (y0, y1) if vertical else (x0, x1)


def overlap(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return max(0.0, hi - lo)


def k_geo(line):
    """Kraken baseline polyline; fall back to the polygon's mid-height chord."""
    pts = (line["baseline"] or {}).get("points") or []
    if len(pts) >= 2:
        return [tuple(p) for p in pts]
    x0, x1, y0, y1 = bbox(line["points"])
    ym = (y0 + y1) / 2
    return [(x0, ym), (x1, ym)]


conn, cur = connect()
cur.execute('select id, "order", image_key, width, height from document_parts where document_id=%s order by "order"', (DOCS[slug],))
parts = cur.fetchall()
summary = []
for part in parts:
    order = part["order"]
    if only is not None and order not in only:
        continue
    name = os.path.basename(part["image_key"])
    stem = os.path.splitext(name)[0]
    cur.execute('select id, "order", baseline, points, manual_geometry from lines where part_id=%s order by "order", created_at', (part["id"],))
    klines = cur.fetchall()
    _, size, regions = parse_page_xml(os.path.join(DATA, slug, "page", stem + ".xml"))

    # XML lines in reading order: regions by readingOrder index (fallback: x-centre), lines in file order
    def ro(r):
        c = r.get("custom", "")
        if "readingOrder" in c and "index:" in c:
            try:
                return (0, int(c.split("index:")[1].split(";")[0]))
            except ValueError:
                pass
        b = bbox(r["coords"]) if r["coords"] else (0, 0, 0, 0)
        return (1, (b[0] + b[1]) / 2)
    xlines = []
    for r in sorted(regions, key=ro):
        for l in r["lines"]:
            geo = l["baseline"] if len(l["baseline"]) >= 2 else (l["coords"] or [])
            if not geo:
                continue
            xlines.append({"id": l["id"], "region": r["id"], "text": l["text"], "geo": [tuple(p) for p in geo],
                           "coords": l["coords"], "vertical": is_vertical(geo)})
    # typical spacing from the biggest region's horizontal baselines
    diffs = []
    for r in regions:
        ys = sorted(statistics.mean(p[1] for p in (l["baseline"] if len(l["baseline"]) >= 2 else l["coords"]))
                    for l in r["lines"] if (l["baseline"] or l["coords"]) and not is_vertical(l["baseline"] if len(l["baseline"]) >= 2 else l["coords"]))
        if len(ys) >= 5:
            diffs += [b - a for a, b in zip(ys, ys[1:], strict=False) if 20 < b - a < 400]
    spacing = statistics.median(diffs) if diffs else 100.0
    thr = 0.4 * spacing

    # score every (K, X)
    cand = {}
    for k in klines:
        kg = k_geo(k)
        kv = is_vertical(kg)
        ks = sample(kg)
        kext = extent(kg, kv)
        klen = max(1.0, kext[1] - kext[0])
        best = []
        for x in xlines:
            if x["vertical"] != kv:
                continue
            xext = extent(x["geo"], kv)
            ov = overlap(kext, xext)
            if ov / klen < 0.5:
                continue
            d = statistics.mean(poly_dist(p, x["geo"]) for p in ks)
            if d < thr:
                best.append((d, ov / max(1.0, xext[1] - xext[0]), x["id"]))
        best.sort()
        cand[str(k["id"])] = best

    # assignment: each K -> best X; each X's primary = K with biggest coverage of X
    by_x = {}
    decisions = {}
    for k in klines:
        kid = str(k["id"])
        o = overrides.get(kid)
        if o == "noise" or o == "keep":
            decisions[kid] = {"decision": o, "x": None, "reason": "override"}
            continue
        if o and o.startswith("pair:"):
            xid = o.split(":", 1)[1]
            by_x.setdefault(xid, []).append((9.0, kid))
            decisions[kid] = {"decision": "pair", "x": xid, "reason": "override"}
            continue
        if not cand[kid]:
            decisions[kid] = {"decision": "noise", "x": None, "reason": f"no xml line within {thr:.0f}px"}
            continue
        d, cov, xid = cand[kid][0]
        by_x.setdefault(xid, []).append((cov, kid))
        decisions[kid] = {"decision": "pair", "x": xid, "reason": f"d={d:.1f} cov={cov:.2f}"}
    xmap = {x["id"]: x for x in xlines}
    for xid, ks in by_x.items():
        ks.sort(reverse=True)
        primary = ks[0][1]
        for _cov, kid in ks[1:]:
            decisions[kid]["decision"] = "split"
            decisions[kid]["reason"] += f" (primary={primary[:8]})"
        if not xmap[xid]["text"]:
            for _, kid in ks:
                decisions[kid]["decision"] = "keep"
                decisions[kid]["reason"] += " xml text empty"
    kmap = {str(k["id"]): k for k in klines}
    # sanity flags: a 'noise' line that is as wide as a real text line deserves a look
    widths = [bbox(kmap[kid]["points"])[1] - bbox(kmap[kid]["points"])[0] for kid, d in decisions.items() if d["decision"] == "pair"]
    med_w = statistics.median(widths) if widths else 0
    for kid, d in decisions.items():
        b = bbox(kmap[kid]["points"])
        d["bbox"] = [round(v) for v in b]
        d["large"] = d["decision"] == "noise" and (b[1] - b[0]) > 0.5 * med_w and not is_vertical(k_geo(kmap[kid]))
        d["manual_geometry"] = kmap[kid]["manual_geometry"]
    primary_of = {d["x"]: kid for kid, d in decisions.items() if d["decision"] == "pair"}
    text_lines = []
    for i, x in enumerate(xlines):
        text_lines.append({"order": i, "xml_id": x["id"], "region": x["region"], "text": x["text"],
                           "paired_line_id": primary_of.get(x["id"]), "geo_bbox": [round(v) for v in bbox(x["geo"])]})
    plan = {"slug": slug, "part_id": str(part["id"]), "order": order, "image": name, "spacing": spacing, "thr": thr,
            "kraken": decisions, "text_lines": text_lines}
    dump_json(plan, os.path.join(SCRATCH, slug, "plan", f"{order:02d}.json"))
    n = lambda dec, decisions=decisions: sum(1 for d in decisions.values() if d["decision"] == dec)
    need_seg = [t for t in text_lines if t["text"] and t["paired_line_id"] is None]
    large = [kid for kid, d in decisions.items() if d["large"]]
    summary.append((order, name, len(klines), n("pair"), n("split"), n("keep"), n("noise"), len(large),
                    sum(1 for t in text_lines if t["text"]), len(need_seg), round(spacing)))
    if RENDER:
        from PIL import Image, ImageDraw, ImageFont
        scale = 0.35
        img = Image.open(os.path.join(SCRATCH, slug, "images", name)).convert("RGB")
        img = img.resize((int(img.width * scale), int(img.height * scale)))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
        except Exception:
            font = ImageFont.load_default()
        sc = lambda pts, scale=scale: [(p[0] * scale, p[1] * scale) for p in pts]
        colors = {"pair": (0, 170, 0), "split": (255, 140, 0), "keep": (150, 0, 200), "noise": (255, 0, 0)}
        for kid, d in decisions.items():
            k = kmap[kid]
            col = (255, 0, 255) if d["large"] else colors[d["decision"]]
            if k["points"]:
                draw.polygon(sc(k["points"]), outline=col)
            draw.line(sc(k_geo(k)), fill=col, width=2)
            x, y = sc(k_geo(k))[0]
            draw.text((x - 24, y - 7), f"k{k['order']}", fill=col, font=font)
        for t in text_lines:
            if t["text"] and t["paired_line_id"] is None:
                x = xmap[t["xml_id"]]
                draw.line(sc(x["geo"]), fill=(0, 0, 255), width=3)
                px, py = sc(x["geo"])[-1]
                draw.text((px + 4, py - 7), f"NEED x{t['order']}", fill=(0, 0, 255), font=font)
        # legend
        draw.rectangle([4, 4, 330, 22], fill=(255, 255, 255))
        draw.text((6, 6), f"p{order} green=pair orange=split purple=keep(empty) red=noise magenta=LARGE noise blue=NEED segment", fill=(0, 0, 0), font=font)
        out = os.path.join(SCRATCH, slug, "plan", f"{order:02d}_{stem}.png")
        img.save(out)
conn.rollback()
print(f"{'pg':>3} {'image':<14} {'kraken':>6} {'pair':>5} {'split':>5} {'keep':>4} {'noise':>5} {'LARGE':>5} {'xml_txt':>7} {'NEED':>4} {'spacing':>7}")
for s in summary:
    print(f"{s[0]:>3} {s[1]:<14} {s[2]:>6} {s[3]:>5} {s[4]:>5} {s[5]:>4} {s[6]:>5} {s[7]:>5} {s[8]:>7} {s[9]:>4} {s[10]:>7}")
print(f"totals: kraken={sum(s[2] for s in summary)} pair={sum(s[3] for s in summary)} split={sum(s[4] for s in summary)} keep={sum(s[5] for s in summary)} noise={sum(s[6] for s in summary)} LARGE={sum(s[7] for s in summary)} xml_txt={sum(s[8] for s in summary)} NEED={sum(s[9] for s in summary)}")
