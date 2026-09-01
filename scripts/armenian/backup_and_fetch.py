"""Back up every segmentation row for both Armenian documents and download the page images."""
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DOCS, SCRATCH, connect, dump_json, fetch_image, media_store

conn, cur = connect()
stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
store = media_store()
for slug, doc_id in DOCS.items():
    cur.execute('select * from document_parts where document_id=%s order by "order"', (doc_id,))
    parts = cur.fetchall()
    part_ids = [p["id"] for p in parts]
    cur.execute("select * from blocks where part_id = any(%s::uuid[])", (part_ids,))
    blocks = cur.fetchall()
    cur.execute("select * from lines where part_id = any(%s::uuid[])", (part_ids,))
    lines = cur.fetchall()
    cur.execute("select lt.* from line_transcriptions lt join lines l on l.id=lt.line_id where l.part_id = any(%s::uuid[])", (part_ids,))
    lts = cur.fetchall()
    cur.execute("select * from page_transcription_lines where part_id = any(%s::uuid[])", (part_ids,))
    ptl = cur.fetchall()
    cur.execute("select * from transcriptions where document_id=%s", (doc_id,))
    layers = cur.fetchall()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{slug}_backup_{stamp}.json")
    dump_json({"document_id": doc_id, "parts": parts, "blocks": blocks, "lines": lines,
               "line_transcriptions": lts, "page_transcription_lines": ptl, "transcriptions": layers}, out)
    print(f"{slug}: parts={len(parts)} blocks={len(blocks)} lines={len(lines)} line_transcriptions={len(lts)} page_text_lines={len(ptl)} layers={[(l['name'], l['kind']) for l in layers]} -> {out}")
    for p in parts:
        name = os.path.basename(p["image_key"])
        dest = os.path.join(SCRATCH, slug, "images", name)
        fetch_image(store, p["image_key"], dest)
    print(f"  images -> {os.path.join(SCRATCH, slug, 'images')} ({len(parts)} files)")
conn.rollback()
