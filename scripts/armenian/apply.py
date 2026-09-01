"""Apply the per-page plans from plan.py to the database for one Armenian document.

usage: apply.py <slug> [--pages 0,3] [--apply]
Dry run by default. Per page: delete 'noise' lines (never manual_geometry ones), renumber the
surviving lines in XML reading order, rewrite page_transcription_lines from the XML text
(paired_line_id = the primary kraken piece), and upsert the ground-truth line_transcriptions,
mirroring TranscriptionService.pair_page_text_line. Idempotent: re-running rewrites the same state.
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DOCS, SCRATCH, connect, load_json

slug = sys.argv[1]
APPLY = "--apply" in sys.argv
only = None
if "--pages" in sys.argv:
    only = {int(v) for v in sys.argv[sys.argv.index("--pages") + 1].split(",")}
conn, cur = connect()
cur.execute("select id from transcriptions where document_id=%s and kind='ground_truth'", (DOCS[slug],))
gt = cur.fetchone()["id"]
print(f"{slug}: ground truth layer {gt}")
tot = dict(deleted=0, kept=0, text=0, paired=0, unpaired=0, skipped_manual=0)
for path in sorted(glob.glob(os.path.join(SCRATCH, slug, "plan", "*.json"))):
    plan = load_json(path)
    if only is not None and plan["order"] not in only:
        continue
    part_id = plan["part_id"]
    cur.execute("select id from lines where part_id=%s", (part_id,))
    db_ids = {str(r["id"]) for r in cur.fetchall()}
    plan_ids = set(plan["kraken"])
    if db_ids != plan_ids:
        print(f"  page {plan['order']}: PLAN STALE (db has {len(db_ids)} lines, plan {len(plan_ids)}); rerun plan.py. Skipping.")
        continue
    noise = [kid for kid, d in plan["kraken"].items() if d["decision"] == "noise" and not d["manual_geometry"]]
    manual_noise = [kid for kid, d in plan["kraken"].items() if d["decision"] == "noise" and d["manual_geometry"]]
    tot["skipped_manual"] += len(manual_noise)
    # reading order of survivors
    sequence = []
    for t in plan["text_lines"]:
        if t["paired_line_id"]:
            sequence.append(t["paired_line_id"])
        splits = [(d["bbox"][0], kid) for kid, d in plan["kraken"].items()
                  if d["decision"] == "split" and d["x"] == t["xml_id"]]
        sequence += [kid for _, kid in sorted(splits)]
    rest = [(d["bbox"][2], kid) for kid, d in plan["kraken"].items()
            if kid not in sequence and (d["decision"] != "noise" or d["manual_geometry"])]
    sequence += [kid for _, kid in sorted(rest)]
    text_lines = [t for t in plan["text_lines"] if t["text"]]
    paired = [t for t in text_lines if t["paired_line_id"]]
    tot["deleted"] += len(noise); tot["kept"] += len(sequence); tot["text"] += len(text_lines)
    tot["paired"] += len(paired); tot["unpaired"] += len(text_lines) - len(paired)
    print(f"  page {plan['order']:>2} {plan['image']:<14} delete={len(noise):<3} keep={len(sequence):<3} text={len(text_lines):<3} paired={len(paired):<3} unpaired={len(text_lines)-len(paired):<3}" + (f" manual-geometry noise kept={len(manual_noise)}" if manual_noise else ""))
    if not APPLY:
        continue
    if noise:
        cur.execute("delete from lines where id = any(%s::uuid[]) and part_id=%s", (noise, part_id))
    for i, kid in enumerate(sequence):
        cur.execute('update lines set "order"=%s where id=%s', (i, kid))
    cur.execute("delete from page_transcription_lines where part_id=%s", (part_id,))
    cur.execute("delete from line_transcriptions where transcription_id=%s and line_id in (select id from lines where part_id=%s)", (gt, part_id))
    for i, t in enumerate(text_lines):
        cur.execute('insert into page_transcription_lines (id, part_id, "order", text, paired_line_id) values (gen_random_uuid(), %s, %s, %s, %s)',
                    (part_id, i, t["text"], t["paired_line_id"]))
        if t["paired_line_id"]:
            cur.execute("insert into line_transcriptions (id, line_id, transcription_id, text, confidence) values (gen_random_uuid(), %s, %s, %s, null) "
                        "on conflict (line_id, transcription_id) do update set text=excluded.text, confidence=null",
                        (t["paired_line_id"], gt, t["text"]))
print(f"totals: {tot}")
if APPLY:
    conn.commit(); print("COMMITTED")
else:
    conn.rollback(); print("DRY RUN - nothing written")
