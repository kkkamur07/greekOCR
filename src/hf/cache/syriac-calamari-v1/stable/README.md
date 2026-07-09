---
language:
- syr
tags:
- handwritten-text-recognition
- calamari
- ocr
library_name: calamari
---

# Syriac HTR (calamari)

Handwritten text recognition checkpoint for **syriac** manuscripts, published from the nomicous **Hub staging tree**.

| Field | Value |
|-------|-------|
| **registry model id** | `syriac-calamari-v1` |
| **registry tag** | `stable` |
| **script** | `syriac` |
| **architecture** | `calamari` |
| **model version** | `v1` |
| **task** | `transcribe` |
| **weights source** | `hf://kkkamur07/syriac-htr-calamari@stable` |

## Usage

Resolve this checkpoint through the inference **Registry** with:

```yaml
weights_source: hf://kkkamur07/syriac-htr-calamari@stable
```

Prefetch into the **Hub cache** without running inference:

```bash
PYTHONPATH=. python scripts/hf/fetch_model.py syriac-calamari-v1 --registry-tag stable
```
