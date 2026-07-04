"""
Here we will be defining the architectures for the model classes we are using i.e. 
1. Calamari OCR : Port from the _support_repo/calamari_ocr
2. TrOCR : HuggingFace Class
3. Kraken : Port from the kraken library or use the adapter. 

Validation requires loading model weights and completing forward passes.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
