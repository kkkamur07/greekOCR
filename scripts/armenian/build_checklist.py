import base64
import html
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SCRATCH, load_json
from PIL import Image, ImageDraw

S = os.path.dirname(SCRATCH)
data = load_json(os.path.join(SCRATCH, 'checklist.json'))
plans = {slug: {pg: load_json(os.path.join(SCRATCH, slug, 'plan', f'{pg:02d}.json')) for pg in range(17)} for slug in data}
_img_cache = {}
def img(slug, name):
    k = (slug, name)
    if k not in _img_cache:
        _img_cache[k] = Image.open(os.path.join(SCRATCH, slug, 'images', name)).convert('RGB')
    return _img_cache[k]

def crop_uri(slug, name, boxes, pad=90, maxw=420):
    """boxes: list of (bbox, colour). Crop around the union, draw outlines, return data URI."""
    im = img(slug, name)
    x0 = min(b[0] for b, _ in boxes); x1 = max(b[1] for b, _ in boxes)
    y0 = min(b[2] for b, _ in boxes); y1 = max(b[3] for b, _ in boxes)
    X0, Y0 = max(0, x0 - pad), max(0, y0 - pad); X1, Y1 = min(im.width, x1 + pad), min(im.height, y1 + pad)
    c = im.crop((X0, Y0, X1, Y1))
    d = ImageDraw.Draw(c)
    for b, col in boxes:
        d.rectangle([b[0] - X0, b[2] - Y0, b[1] - X0, b[3] - Y0], outline=col, width=4)
    if c.width > maxw:
        c = c.resize((maxw, int(c.height * maxw / c.width)))
    buf = io.BytesIO(); c.save(buf, 'JPEG', quality=72)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()

def where(slug, bbox, size):
    w, h = size
    cx = (bbox[0] + bbox[1]) / 2; cy = (bbox[2] + bbox[3]) / 2
    zones = [(0.19, 'left margin'), (0.53, 'left column'), (0.575, 'gutter'), (0.91, 'right column'), (9, 'right margin')] if slug == 'ms_p_172' else \
            [(0.22, 'left margin'), (0.555, 'left column'), (0.615, 'gutter'), (0.95, 'right column'), (9, 'right margin')]
    col = next(lab for lim, lab in zones if cx / w < lim)
    if cy / h < 0.1: col = 'top / header'
    if cy / h > 0.93: col = 'bottom margin'
    return f"{col}, {round(100 * cy / h)}% down"

sizes = {'ms_p_172': (2479, 3508), 'ms_ucla': None}
items_html = {}
totals = {}
for slug, d in data.items():
    pages_html = []
    tot = dict(draw=0, merge=0, untranscribed=0, text=0, paired=0, lines=0)
    for p in d['pages']:
        plan = plans[slug][p['order']]
        im = img(slug, p['image']); size = (im.width, im.height)
        paired_boxes = [pd['bbox'] for pd in plan['kraken'].values() if pd['decision'] == 'pair']
        rows = []
        for n in p['need']:
            b = n['bbox']; cx = (b[0] + b[1]) / 2; cy = (b[2] + b[3]) / 2
            inside = [pb for pb in paired_boxes if pb[0] <= cx <= pb[1] and pb[2] <= cy <= pb[3]]
            note = 'This numeral sits inside a segment that is already paired to the line above it: split that segment (or draw a new one over the numeral), then pair.' if inside else 'No segment covers this text. Draw one around it, then pair it in the pairing panel.'
            boxes = [(b, (0, 90, 220))] + [(pb, (0, 150, 60)) for pb in inside[:1]]
            rows.append(('draw', n['text'], f"text line {n['order'] + 1}", where(slug, b, size), note, crop_uri(slug, p['image'], boxes, pad=120 if not inside else 60)))
        for s in p['splits']:
            boxes = [(s['bbox'], (235, 120, 0))] + ([(s['primary_bbox'], (0, 150, 60))] if s['primary_bbox'] else [])
            rows.append(('merge', s['text'], f"text line {s['order'] + 1}", where(slug, s['bbox'], size), 'Kraken cut this line in two. The green segment already carries the text; merge the orange piece into it, or delete the orange piece if it is only a stray fragment.', crop_uri(slug, p['image'], boxes, pad=60)))
        for k in p['keeps']:
            rows.append(('untranscribed', k['xml_text'] or '', '', where(slug, k['bbox'], size), 'This segment has ink but no transcription in the XML (the XML line is empty or missing). Type the text in the editor, or delete the segment if it should not be trained on.', crop_uri(slug, p['image'], [(k['bbox'], (150, 0, 200))], pad=60)))
        for r in rows: tot[r[0]] += 1
        tot['text'] += p['text']; tot['paired'] += p['paired']; tot['lines'] += p['lines']
        pct = round(100 * p['paired'] / p['text']) if p['text'] else 0
        rows_html = ''.join(
            f'<li class="item {kind}"><label><input type="checkbox" data-key="{slug}-{p["order"]}-{i}"><span class="kind">{ {"draw":"Draw","merge":"Merge","untranscribed":"Transcribe"}[kind]}</span>'
            f'<span class="txt" lang="hy">{html.escape(text) or "<em>no text</em>"}</span><span class="meta">{html.escape(ref)}{" · " if ref else ""}{html.escape(loc)}</span></label>'
            f'<img src="{uri}" alt=""><p class="note">{html.escape(note)}</p></li>'
            for i, (kind, text, ref, loc, note, uri) in enumerate(rows))
        state = 'done' if not rows else ''
        pages_html.append(
            f'<section class="page {state}" id="{slug}-p{p["page"]}"><header><h3>Page {p["page"]} <span class="img">{html.escape(p["image"])}</span></h3>'
            f'<div class="stats"><span><b>{p["paired"]}</b> / {p["text"]} text lines paired</span><span><b>{p["lines"]}</b> segments</span>'
            f'<span class="bar"><i style="width:{pct}%"></i></span><span class="todo">{len(rows) or "nothing"} to do</span></div></header>'
            + (f'<ul class="items">{rows_html}</ul>' if rows else '<p class="clean">Every text line on this page is paired. Nothing to do here.</p>') + '</section>')
    items_html[slug] = ''.join(pages_html); totals[slug] = tot

def summary(slug):
    t = totals[slug]; title = data[slug]['title']
    return (f'<div class="sum" data-slug="{slug}"><h2>{title}</h2><dl><div><dt>Paired</dt><dd>{t["paired"]} <small>/ {t["text"]} text lines</small></dd></div>'
            f'<div><dt>Segments now</dt><dd>{t["lines"]}</dd></div><div class="draw"><dt>Draw</dt><dd>{t["draw"]}</dd></div>'
            f'<div class="merge"><dt>Merge</dt><dd>{t["merge"]}</dd></div><div class="untranscribed"><dt>Transcribe</dt><dd>{t["untranscribed"]}</dd></div></dl></div>')

page = f'''<title>Armenian Pairing Worklist</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Noto+Serif+Armenian:wght@500&display=swap">
<style>
:root {{ --bg:#f2f3f0; --card:#ffffff; --ink:#1f2224; --muted:#5f6569; --rule:#d9dcd6; --accent:#a3312a; --accent-ink:#ffffff;
  --draw:#0b5cc4; --draw-bg:#e8f0fb; --merge:#b85c00; --merge-bg:#fbeedd; --untr:#6a2fa0; --untr-bg:#f0e8f8; --good:#2c7a3f; --good-bg:#e4f1e6; --focus:#0b5cc4; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#191a19; --card:#232524; --ink:#e9e6e0; --muted:#a4a8a3; --rule:#3a3d3a; --accent:#e2695f; --accent-ink:#1a1a1a;
  --draw:#7fb2f5; --draw-bg:#1c2a3d; --merge:#f0a250; --merge-bg:#3a2a16; --untr:#c39af0; --untr-bg:#2e2140; --good:#7fcf92; --good-bg:#1d3324; --focus:#7fb2f5; }} }}
:root[data-theme="dark"] {{ --bg:#191a19; --card:#232524; --ink:#e9e6e0; --muted:#a4a8a3; --rule:#3a3d3a; --accent:#e2695f; --accent-ink:#1a1a1a;
  --draw:#7fb2f5; --draw-bg:#1c2a3d; --merge:#f0a250; --merge-bg:#3a2a16; --untr:#c39af0; --untr-bg:#2e2140; --good:#7fcf92; --good-bg:#1d3324; --focus:#7fb2f5; }}
body {{ background:var(--bg); color:var(--ink); font:15px/1.5 "IBM Plex Sans", system-ui, sans-serif; font-variant-numeric: tabular-nums; }}
main {{ max-width: 1080px; margin: 0 auto; padding: 32px 24px 80px; }}
h1 {{ font-size: 28px; font-weight: 600; margin: 0 0 4px; letter-spacing: -0.01em; text-wrap: balance; }}
.lede {{ color: var(--muted); max-width: 68ch; margin: 0 0 24px; }}
.legend {{ display:flex; flex-wrap:wrap; gap: 10px 18px; margin: 0 0 28px; font-size: 13px; color: var(--muted); }}
.legend b {{ display:inline-block; padding: 1px 8px; border-radius: 3px; font-weight: 600; margin-right: 6px; }}
.tabs {{ display:flex; gap: 8px; border-bottom: 1px solid var(--rule); margin-bottom: 20px; }}
.tabs button {{ background: none; border: 0; border-bottom: 3px solid transparent; color: var(--muted); font: inherit; font-weight: 600; padding: 8px 4px 10px; cursor: pointer; }}
.tabs button[aria-selected="true"] {{ color: var(--ink); border-bottom-color: var(--accent); }}
.tabs button:focus-visible, .item input:focus-visible {{ outline: 2px solid var(--focus); outline-offset: 2px; }}
.sum h2 {{ font-size: 20px; margin: 0 0 10px; }}
.sum dl {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin: 0 0 28px; }}
.sum dl > div {{ background: var(--card); border: 1px solid var(--rule); border-radius: 6px; padding: 12px 14px; }}
.sum dt {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }}
.sum dd {{ margin: 2px 0 0; font-size: 24px; font-weight: 600; }} .sum dd small {{ font-size: 13px; font-weight: 400; color: var(--muted); }}
.sum .draw dd {{ color: var(--draw); }} .sum .merge dd {{ color: var(--merge); }} .sum .untranscribed dd {{ color: var(--untr); }}
.page {{ background: var(--card); border: 1px solid var(--rule); border-radius: 8px; margin: 0 0 16px; overflow: hidden; }}
.page header {{ display:flex; flex-wrap: wrap; align-items: baseline; gap: 6px 20px; padding: 14px 18px; border-bottom: 1px solid var(--rule); }}
.page.done header {{ border-bottom: 0; }}
.page h3 {{ margin: 0; font-size: 17px; }} .page h3 .img {{ font-weight: 400; color: var(--muted); font-size: 13px; margin-left: 8px; }}
.stats {{ display:flex; flex-wrap: wrap; align-items: center; gap: 6px 16px; font-size: 13px; color: var(--muted); margin-left: auto; }}
.stats b {{ color: var(--ink); }}
.bar {{ width: 110px; height: 6px; background: var(--rule); border-radius: 3px; overflow: hidden; }} .bar i {{ display:block; height: 100%; background: var(--good); }}
.todo {{ font-weight: 600; color: var(--ink); }}
.clean {{ margin: 0; padding: 12px 18px; color: var(--good); font-size: 14px; }}
.items {{ list-style: none; margin: 0; padding: 6px 18px 14px; display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 12px; }}
.item {{ border: 1px solid var(--rule); border-radius: 6px; padding: 10px 12px; display: flex; flex-direction: column; gap: 8px; background: var(--bg); }}
.item.checked {{ opacity: 0.45; }}
.item label {{ display: grid; grid-template-columns: auto auto 1fr; align-items: center; column-gap: 10px; cursor: pointer; }}
.item label .meta {{ grid-column: 2 / 4; font-size: 12px; color: var(--muted); }}
.item input {{ width: 18px; height: 18px; margin: 0; accent-color: var(--accent); }}
.kind {{ font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; padding: 2px 7px; border-radius: 3px; }}
.draw .kind, .legend .draw {{ color: var(--draw); background: var(--draw-bg); }} .merge .kind, .legend .merge {{ color: var(--merge); background: var(--merge-bg); }} .untranscribed .kind, .legend .untranscribed {{ color: var(--untr); background: var(--untr-bg); }} .legend .good {{ color: var(--good); background: var(--good-bg); }}
.txt {{ font: 500 20px/1.3 "Noto Serif Armenian", "Noto Serif", serif; }} .txt em {{ font: italic 14px "IBM Plex Sans", sans-serif; color: var(--muted); }}
.item img {{ width: 100%; height: auto; border-radius: 4px; border: 1px solid var(--rule); background: #fff; }}
.note {{ margin: 0; font-size: 13px; color: var(--muted); }}
.panel[hidden] {{ display: none; }}
@media (prefers-reduced-motion: no-preference) {{ .item {{ transition: opacity .15s; }} }}
</style>
<main>
<h1>Armenian Pairing Worklist</h1>
<p class="lede">Everything the automatic pass could not settle on MS_P_172 and MS_UCLA_MS, page by page. Each card shows a crop of the spot on the page. Tick items off as you go; ticks are remembered in this browser only.</p>
<div class="legend"><span><b class="draw">Draw</b> text line with no segment: draw one, then pair</span><span><b class="merge">Merge</b> orange fragment to merge into the green segment (or delete)</span><span><b class="untranscribed">Transcribe</b> segment with ink but no text</span><span><b class="good">Green box</b> already paired segment, for reference</span></div>
<div class="tabs" role="tablist"><button role="tab" aria-selected="true" data-tab="ms_p_172">MS_P_172</button><button role="tab" aria-selected="false" data-tab="ms_ucla">MS_UCLA_MS</button></div>
<div class="panel" id="panel-ms_p_172">{summary('ms_p_172')}{items_html['ms_p_172']}</div>
<div class="panel" id="panel-ms_ucla" hidden>{summary('ms_ucla')}{items_html['ms_ucla']}</div>
</main>
<script>
(function(){{
  var tabs = document.querySelectorAll('.tabs button');
  function show(slug){{ tabs.forEach(function(b){{ b.setAttribute('aria-selected', b.dataset.tab===slug); }}); document.querySelectorAll('.panel').forEach(function(p){{ p.hidden = p.id !== 'panel-'+slug; }}); try {{ localStorage.setItem('arm-tab', slug); }} catch(e) {{}} }}
  tabs.forEach(function(b){{ b.addEventListener('click', function(){{ show(b.dataset.tab); }}); }});
  var saved = null; try {{ saved = localStorage.getItem('arm-tab'); }} catch(e) {{}}
  if (saved === 'ms_ucla') show(saved);
  document.querySelectorAll('.item input').forEach(function(cb){{
    var key = 'arm-' + cb.dataset.key; var on = false; try {{ on = localStorage.getItem(key) === '1'; }} catch(e) {{}}
    cb.checked = on; cb.closest('.item').classList.toggle('checked', on);
    cb.addEventListener('change', function(){{ cb.closest('.item').classList.toggle('checked', cb.checked); try {{ localStorage.setItem(key, cb.checked ? '1' : '0'); }} catch(e) {{}} }});
  }});
}})();
</script>
'''
out = os.path.join(S, 'armenian_pairing_worklist.html')
with open(out, 'w') as handle:
    handle.write(page)
print(out, round(os.path.getsize(out) / 1e6, 2), 'MB', totals)
