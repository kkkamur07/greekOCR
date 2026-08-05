# Nomicous landing page

Minimal static site for **nomicous.com**. Same palette as the app (navy + warm neutrals). Will be expanded later.

## Preview

```bash
cd landing && python -m http.server 8080
```

Open http://localhost:8080

## Deploy

Upload the `landing/` folder to any static host, or use the Vercel project in [`docs/deployment/production.md`](../docs/deployment/production.md). Point **nomicous.com** here; the annotation app lives at **app.nomicous.com**.

## Edit

- Copy and links: `index.html`
- Styles: `styles.css` (tokens match `nomicous/frontend/src/styles/theme-shell.css`)
- Screenshots: `assets/screenshots/` (responsive `.webp` at 640 / 960 / 1280px)

Re-encode after replacing a source capture (save as `*-source.png`, then):

```bash
cd landing/assets/screenshots
for f in editor live-view; do
  for w in 640 960 1280; do
    cwebp -q 82 -resize "$w" 0 "${f}-source.png" -o "${f}-${w}.webp"
  done
done
```

Layout is mobile-first: the hero splits into copy plus screenshot from 56rem up, and the live-view shot spans the full `--wide` column. Nav links stay visible at all widths (44px touch targets).
