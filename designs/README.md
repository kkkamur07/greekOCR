# Nomicous UI designs

Minimal static HTML mockups for the full platform — navy + warm neutrals, no blue tint.

## View locally

```bash
python3 -m http.server 8765 --directory designs
# http://localhost:8765
```

Or open `designs/index.html` directly.

## Screens

| File | Description |
|------|-------------|
| `nomicous/login.html` | Sign in |
| `nomicous/register.html` | Register |
| `nomicous/projects.html` | Project list |
| `nomicous/project-dashboard.html` | Documents in a project |
| `nomicous/document-detail.html` | Parts, upload, jobs |
| `nomicous/page-editor-canvas.html` | Full canvas (default) — strip appears on segment select |
| `nomicous/page-editor.html` | Canvas + resizable PDF pane + confidence strip |
| `nomicous/public-document.html` | Public read-only view |

## Styles

- `nomicous/theme.css` — shared tokens, nav, lists, auth
- `nomicous/page-editor.css` — editor shell, toolbar, canvas, drawer
- `assets/nomos.svg` — brand mark
