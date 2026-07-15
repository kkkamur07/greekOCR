1. Verification of the whole codebase including the tests
2. Development of the local helpers - like more ui side so that they are convenient to use.
3. Move off Kraken for segmentation: port the BLLA/segment model + obtain/publish weights under our registry (same pattern as Calamari). Goal — drop `kraken` (and its click/torch constraint drag) from the inference runtime so the helper and inference image stay lighter, and we can clear the Click/Torch pip-audit VEX entries once replacements land.
4. After (3): remove Click VEX (`docs/security/vex-click-pysec-2026-2132.md`) and Torch VEX (`docs/security/vex-torch-pysec-2026-139-cve-2025-3000.md`) when deps no longer pull vulnerable click/torch floors.
