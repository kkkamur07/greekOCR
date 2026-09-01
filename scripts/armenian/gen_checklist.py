import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SCRATCH, connect, dump_json, load_json

conn, cur = connect()
out = {}
for slug, title in (('ms_p_172','MS_P_172'), ('ms_ucla','MS_UCLA_MS')):
    pages = []
    for pg in range(17):
        plan = load_json(os.path.join(SCRATCH,slug,'plan',f'{pg:02d}.json'))
        cur.execute("select count(*) as n from lines where part_id=%s", (plan['part_id'],)); nlines = cur.fetchone()['n']
        cur.execute("select paired_line_id, text from page_transcription_lines where part_id=%s", (plan['part_id'],))
        db_text = cur.fetchall()
        paired = sum(1 for r in db_text if r['paired_line_id'])
        paired_texts = {r['text'] for r in db_text if r['paired_line_id']}
        xmap = {t['xml_id']: t for t in plan['text_lines']}
        # a text line counts as needing work only if the DB panel still shows it unpaired
        need = [dict(order=t['order'], text=t['text'], bbox=t['geo_bbox']) for t in plan['text_lines'] if t['text'] and not t['paired_line_id'] and t['text'] not in paired_texts]
        splits = []
        for _kid, d in plan['kraken'].items():
            if d['decision'] == 'split':
                prim = [pk for pk, pd in plan['kraken'].items() if pd['decision']=='pair' and pd['x']==d['x']]
                splits.append(dict(bbox=d['bbox'], primary_bbox=plan['kraken'][prim[0]]['bbox'] if prim else None, text=xmap[d['x']]['text'], order=xmap[d['x']]['order']))
        keeps = [dict(bbox=d['bbox'], xml_text=(xmap[d['x']]['text'] if d['x'] else None)) for kid, d in plan['kraken'].items() if d['decision']=='keep']
        pages.append(dict(page=pg+1, order=pg, image=plan['image'], lines=nlines, text=len(db_text), paired=paired, need=need, splits=splits, keeps=keeps))
    out[slug] = dict(title=title, pages=pages)
conn.rollback()
dump_json(out, os.path.join(SCRATCH,'checklist.json'))
for slug, d in out.items():
    print(slug, 'need', sum(len(p['need']) for p in d['pages']), 'splits', sum(len(p['splits']) for p in d['pages']), 'keeps', sum(len(p['keeps']) for p in d['pages']))
