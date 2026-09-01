"""Per-page DB status for one Armenian document: lines, paired lines, text lines, unpaired text. Read-only."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DOCS, connect

slug = sys.argv[1]
conn, cur = connect()
cur.execute("""
    select pt."order", pt.image_key,
      (select count(*) from lines l where l.part_id=pt.id) as lines,
      (select count(*) from lines l where l.part_id=pt.id and l.manual_geometry) as manual,
      (select count(*) from page_transcription_lines t where t.part_id=pt.id) as text_lines,
      (select count(*) from page_transcription_lines t where t.part_id=pt.id and t.paired_line_id is not null) as paired,
      (select count(*) from line_transcriptions lt join lines l on l.id=lt.line_id join transcriptions tr on tr.id=lt.transcription_id
         where l.part_id=pt.id and tr.kind='ground_truth' and lt.text<>'') as gt
    from document_parts pt where pt.document_id=%s order by pt."order" """, (DOCS[slug],))
rows = cur.fetchall(); conn.rollback()
print(f"{'pg':>3} {'image':<14} {'lines':>5} {'manual':>6} {'text':>5} {'paired':>6} {'gt':>4} {'unpaired_text':>13} {'unpaired_seg':>12}")
for r in rows:
    print(f"{r['order']:>3} {os.path.basename(r['image_key']):<14} {r['lines']:>5} {r['manual']:>6} {r['text_lines']:>5} {r['paired']:>6} {r['gt']:>4} {r['text_lines']-r['paired']:>13} {r['lines']-r['paired']:>12}")
print(f"totals: lines={sum(r['lines'] for r in rows)} text={sum(r['text_lines'] for r in rows)} paired={sum(r['paired'] for r in rows)} gt={sum(r['gt'] for r in rows)}")
