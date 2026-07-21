1. Verification of the whole codebase including the tests
2. Development of the local helpers - like more ui side so that they are convenient to use.
3. ✅ Move off the Kraken package for segmentation: native BLLA PyTorch
   topology, preprocessing, decoder, registry-pinned safetensors, and real-page
   parity coverage are complete. The optional `parity` dependency group remains
   only as a development oracle.
4. Remove the Click VEX
   (`docs/security/vex-click-pysec-2026-2132.md`) when the inference dependency
   graph resolves Click >=8.3.3, and remove the Torch VEX
   (`docs/security/vex-torch-pysec-2026-139-cve-2025-3000.md`) when its
   remaining PyTorch floor no longer requires it.
5. Safari stability (app.nomicous.com + api.nomicous.com):
   - **Session reload / 403 on `/auth/refresh`:** Arc/Chrome keep the session after reload; Safari often returns CSRF `403`. Root cause is cross-subdomain cookie auth - session is `__Host-` on `api.nomicous.com`, CSRF (`greekocr-csrf`, `Domain=.nomicous.com`) must be readable by JS on `app.nomicous.com` for `X-CSRF-Token`. Safari ITP is stricter about that sibling-subdomain cookie than Chromium. Durable fix: same-origin BFF/proxy on `app.nomicous.com` so cookies are first-party to the app host.
   - **Local inference "not connected":** HTTPS page probing `http://127.0.0.1:8001` - Chrome/Arc allow loopback; Safari often blocks mixed content / local-network access. Verify helper CORS includes `https://app.nomicous.com`, then either document Safari Local Network permission or serve a loopback TLS helper / proxy path.
   - **Transcription PDF inline preview:** Safari frequently cannot embed `blob:` PDFs in `<object>`/`<iframe>`; Chrome/Arc usually can. Longer-term option is PDF.js for a consistent in-app viewer.
   - **Verify in Safari Web Inspector:** Storage cookies on both hosts; Network on `/auth/refresh` (cookies + `X-CSRF-Token`); Console for helper `/health` and PDF embed failures.
6. Persist cancelled **local** helper jobs into project job history (today only successful local runs call `record_local_job` with `done`; cancel stays in-memory only, so the project Jobs empty state cannot show them).
8. ✅ Add the helper for macOS Intel and test the helper on other computers and platforms as well.
9. ✅ Inference Helper URL fallback: try the provided URL, IPv4, IPv6, and `localhost:8001`; add Ubuntu diagnostics for connection-refused failures.
