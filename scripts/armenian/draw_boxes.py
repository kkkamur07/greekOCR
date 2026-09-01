"""Draw rectangular segments for the still-unpaired XML text lines of one Armenian document and pair them.

usage: draw_boxes.py <slug> [--apply]
For each page_transcription_lines row with paired_line_id NULL whose XML line has Coords, insert a manual
polygon line (4-corner box around the XML coords, padded), pair it, and write the ground-truth text.
Dry run renders every proposed box to SCRATCH/<slug>/draw/sheet.png.
"""
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psycopg2.extras
from common import DATA, DOCS, SCRATCH, bbox, connect, parse_page_xml

slug = sys.argv[1]; APPLY = "--apply" in sys.argv; PAD = 8; MIN_H = 110; MIN_W = 80
conn, cur = connect()
cur.execute("select id from transcriptions where document_id=%s and kind='ground_truth'", (DOCS[slug],)); gt = cur.fetchone()["id"]
cur.execute('select id, "order", image_key, width, height from document_parts where document_id=%s order by "order"', (DOCS[slug],))
parts = cur.fetchall()
jobs = []
for part in parts:
    stem = os.path.splitext(os.path.basename(part["image_key"]))[0]
    _, size, regions = parse_page_xml(os.path.join(DATA, slug, "page", stem + ".xml"))
    xml_by_text = {}
    for r in regions:
        for l in r["lines"]:
            if l["text"] and l["coords"]:
                xml_by_text.setdefault(l["text"], []).append(l)
    cur.execute('select coalesce(max("order"), -1) as m from lines where part_id=%s', (part["id"],)); next_order = cur.fetchone()["m"] + 1
    cur.execute('select id, "order", text from page_transcription_lines where part_id=%s and paired_line_id is null order by "order"', (part["id"],))
    for t in cur.fetchall():
        cands = xml_by_text.get(t["text"], [])
        if len(cands) != 1:
            print(f"page {part['order']+1}: text line {t['order']+1} '{t['text']}' has {len(cands)} XML matches, skipped"); continue
        x0, x1, y0, y1 = bbox(cands[0]["coords"])
        if (y1 - y0) < MIN_H:
            cy = (y0 + y1) / 2; y0, y1 = cy - MIN_H / 2, cy + MIN_H / 2
        if (x1 - x0) < MIN_W:
            cx = (x0 + x1) / 2; x0, x1 = cx - MIN_W / 2, cx + MIN_W / 2
        x0, y0 = max(0, x0 - PAD), max(0, y0 - PAD); x1, y1 = min(part["width"], x1 + PAD), min(part["height"], y1 + PAD)
        box = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        by = y1 - 0.2 * (y1 - y0)
        jobs.append(dict(part=part, text_line=t, box=box, baseline=[[x0, by], [x1, by]], order=next_order)); next_order += 1
print(f"{slug}: {len(jobs)} boxes to draw")
for j in jobs:
    b = bbox(j["box"]); print(f"  page {j['part']['order']+1:>2} text line {j['text_line']['order']+1:>2} box=[{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}] {b[1]-b[0]:.0f}x{b[3]-b[2]:.0f} | {j['text_line']['text']}")
if APPLY:
    for j in jobs:
        lid = uuid.uuid4()
        cur.execute('insert into lines (id, part_id, block_id, baseline, mask, kind, points, source, source_metadata, kraken_ceiling, manual_geometry, "order") '
                    "values (%s, %s, (select id from blocks where part_id=%s order by \"order\" limit 1), %s, %s, 'polygon', %s, 'manual', %s, null, true, %s)",
                    (lid, j["part"]["id"], j["part"]["id"], psycopg2.extras.Json({"points": j["baseline"]}), psycopg2.extras.Json({"points": j["box"]}),
                     psycopg2.extras.Json(j["box"]), psycopg2.extras.Json({"drawn_by": "scripts/armenian/draw_boxes.py", "from_xml_coords": True}), j["order"]))
        cur.execute("update page_transcription_lines set paired_line_id=%s where id=%s and paired_line_id is null", (lid, j["text_line"]["id"]))
        cur.execute("insert into line_transcriptions (id, line_id, transcription_id, text, confidence) values (gen_random_uuid(), %s, %s, %s, null)", (lid, gt, j["text_line"]["text"]))
    conn.commit(); print("COMMITTED")
else:
    conn.rollback()
    from PIL import Image, ImageDraw
    tiles = []
    for j in jobs:
        im = Image.open(os.path.join(SCRATCH, slug, "images", os.path.basename(j["part"]["image_key"]))).convert("RGB")
        x0, x1, y0, y1 = bbox(j["box"]); pad = 70
        c = im.crop((max(0, x0 - pad), max(0, y0 - pad), min(im.width, x1 + pad), min(im.height, y1 + pad)))
        d = ImageDraw.Draw(c); ox, oy = max(0, x0 - pad), max(0, y0 - pad)
        d.rectangle([x0 - ox, y0 - oy, x1 - ox, y1 - oy], outline=(0, 90, 220), width=3)
        d.line([(x0 - ox, j["baseline"][0][1] - oy), (x1 - ox, j["baseline"][0][1] - oy)], fill=(0, 90, 220), width=2)
        c = c.resize((280, int(c.height * 280 / c.width))); tiles.append(c)
    if not tiles:
        print("DRY RUN - nothing written; no boxes proposed")
        sys.exit(0)
    cols = 6; rows = (len(tiles) + cols - 1) // cols; th = max(t.height for t in tiles)
    sheet = Image.new("RGB", (cols * 286, rows * (th + 6)), (255, 255, 255))
    for i, t in enumerate(tiles): sheet.paste(t, ((i % cols) * 286, (i // cols) * (th + 6)))
    os.makedirs(os.path.join(SCRATCH, slug, "draw"), exist_ok=True); out = os.path.join(SCRATCH, slug, "draw", "sheet.png"); sheet.save(out)
    print("DRY RUN - nothing written; sheet:", out)
