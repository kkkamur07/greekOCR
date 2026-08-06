# Internal backlog

1. Verify the whole codebase, tests included.

2. ~~Develop the local helpers, more on the UI side, so they are convenient to use.~~
   Retired by ADR 0002 and #60. Local inference is a CLI, and its interface is the
   terminal plus the one account setting, so there is no helper UI to develop.

3. Done: ~~move off the Kraken package for segmentation.~~ The inference-owned BLLA
   topology, preprocessing, decoder, and registry-pinned artifact are all complete.
   The `parity` dependency group is still gone: ADR 0006 brought the ONNX runtime
   back, but Kraken is not the oracle any more - the Torch graph in
   `src/model/inference_export/` is, and it is already in the repository.

4. Remove the Click VEX (`docs/security/vex-click-pysec-2026-2132.md`) once the
   inference dependency graph resolves Click >=8.3.3. The Torch VEX
   (`docs/security/vex-torch-pysec-2026-139-cve-2025-3000.md`) no longer covers
   anything a researcher installs - ADR 0006 took Torch out of the published
   closure - but it still covers `--group export` on a maintainer's machine, so
   it stays until nothing in the repository needs Torch.

5. Safari stability on app.nomicous.com and api.nomicous.com:

   - Session reload returns a CSRF `403` on `/auth/refresh`. Arc and Chrome keep the
     session after a reload; Safari often does not. The cause is cross-subdomain cookie
     auth: the session is `__Host-` on `api.nomicous.com`, while the CSRF cookie
     (`greekocr-csrf`, `Domain=.nomicous.com`) has to be readable by JS on
     `app.nomicous.com` to populate `X-CSRF-Token`. Safari ITP treats that
     sibling-subdomain cookie more strictly than Chromium does. The durable fix is a
     same-origin BFF or proxy on `app.nomicous.com`, so the cookies are first-party to
     the app host.
   - ~~Local inference reports "not connected" because an HTTPS page is probing
     `http://127.0.0.1:8001`.~~ Fixed by deletion in #60. The page no longer probes
     anything: the agent connects out to the platform, so no browser's local-network
     policy sits on the path. This was the failure ADR 0002 was written about.
   - Safari frequently cannot embed `blob:` PDFs in `<object>` or `<iframe>` for the
     transcription inline preview, where Chrome and Arc usually can. The longer-term
     option is PDF.js, for a viewer that behaves the same everywhere.
   - To verify in Safari Web Inspector: check Storage for cookies on both hosts, Network
     for `/auth/refresh` (cookies and `X-CSRF-Token`), and Console for helper `/health`
     and PDF embed failures.

6. ~~Persist cancelled local helper jobs into project job history.~~ Moot since #60. A
   local run is an ordinary platform job now, so cancelling one is recorded exactly the
   way cancelling any other job is.

7. Done: ~~add the helper for macOS Intel, and test the helper on other computers and
   platforms.~~ Superseded by #60 and #61. There is now one published package for every
   platform.

8. Done: ~~inference helper URL fallback, trying the provided URL, then IPv4, then IPv6,
   then `localhost:8001`.~~ Deleted along with the transport in #60. There is no URL left
   to fall back through.
