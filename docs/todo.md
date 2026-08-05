1. Verification of the whole codebase including the tests
2. ~~Development of the local helpers - like more ui side so that they are convenient to use.~~
   Retired by ADR 0002 / #60: local inference is a CLI, and its interface is
   the terminal plus the one account setting. There is no helper UI to develop.
3. ✅ Move off the Kraken package for segmentation: native BLLA PyTorch
   topology, preprocessing, decoder, and registry-pinned safetensors are
   complete. The `parity` dependency group is gone with the ONNX runtime it
   compared against (ADR 0004).
4. Remove the Click VEX
   (`docs/security/vex-click-pysec-2026-2132.md`) when the inference dependency
   graph resolves Click >=8.3.3, and remove the Torch VEX
   (`docs/security/vex-torch-pysec-2026-139-cve-2025-3000.md`) when its
   remaining PyTorch floor no longer requires it.
5. Safari stability (app.nomicous.com + api.nomicous.com):
   - **Session reload / 403 on `/auth/refresh`:** Arc/Chrome keep the session after reload; Safari often returns CSRF `403`. Root cause is cross-subdomain cookie auth - session is `__Host-` on `api.nomicous.com`, CSRF (`greekocr-csrf`, `Domain=.nomicous.com`) must be readable by JS on `app.nomicous.com` for `X-CSRF-Token`. Safari ITP is stricter about that sibling-subdomain cookie than Chromium. Durable fix: same-origin BFF/proxy on `app.nomicous.com` so cookies are first-party to the app host.
   - ~~**Local inference "not connected":** HTTPS page probing `http://127.0.0.1:8001`.~~
     Fixed by deletion (#60). The page no longer probes anything: the agent
     connects out to the platform, so no browser's local-network policy is on
     the path. This was the failure ADR 0002 was written about.
   - **Transcription PDF inline preview:** Safari frequently cannot embed `blob:` PDFs in `<object>`/`<iframe>`; Chrome/Arc usually can. Longer-term option is PDF.js for a consistent in-app viewer.
   - **Verify in Safari Web Inspector:** Storage cookies on both hosts; Network on `/auth/refresh` (cookies + `X-CSRF-Token`); Console for helper `/health` and PDF embed failures.
6. ~~Persist cancelled **local** helper jobs into project job history.~~
   Moot since #60: a local run is an ordinary platform job now, so cancelling
   one is recorded the same way cancelling any other job is.
8. ✅ ~~Add the helper for macOS Intel and test the helper on other computers and platforms as well.~~ Superseded: there is one **published package** for every platform (#60, #61).
9. ✅ ~~Inference Helper URL fallback: try the provided URL, IPv4, IPv6, and `localhost:8001`.~~ Deleted with the transport (#60); there is no URL to fall back through.
