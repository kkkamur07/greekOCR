"""
Document binarization.

The backend (`POST /api/binarize`) uses Kraken ``nlbin``. Use the same path here for
matching results. If Kraken is not installed, a lighter OpenCV fallback runs.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_INPUT = _REPO_ROOT / "backend/uploads/c9378fd7-d934-48d3-aadb-6d5917ff90a7.png"


def _binarize_nlbin(image_path: str | Path, output_path: str | Path) -> None:
    """Same algorithm as ``backend/app/main.py`` (Kraken nonlinear binarization)."""
    from PIL import Image
    from kraken import binarization

    im = Image.open(image_path)
    out = binarization.nlbin(im)
    out.save(output_path)


def sauvola_threshold_map(
    img: np.ndarray,
    window_size: int = 45,
    k: float = 0.34,
    r: float | None = None,
) -> np.ndarray:
    """
    Sauvola adaptive threshold surface: T = m * (1 + k * ((s/R) - 1)).
    Often beats Niblack on uneven parchment / scans.
    """
    x = np.asarray(img, dtype=np.float64)
    if r is None:
        r = max(float(np.ptp(x)), 1.0)
    w = window_size if window_size % 2 == 1 else window_size + 1
    ksum = float(w * w)
    kernel = np.ones((w, w), dtype=np.float64)
    m = cv2.filter2D(x, -1, kernel, borderType=cv2.BORDER_REFLECT) / ksum
    m2 = cv2.filter2D(x * x, -1, kernel, borderType=cv2.BORDER_REFLECT) / ksum
    s = np.sqrt(np.maximum(m2 - m * m, 0.0))
    return m * (1.0 + k * ((s / r) - 1.0))


def _binarize_local_fallback(image_path: str | Path, output_path: str | Path) -> None:
    """
    OpenCV-only pipeline: bilateral smoothing + Sauvola + light morphology.
    Output polarity matches Kraken (white strokes on black background).
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(image_path))

    # Edge-preserving smoothing (fixed diameter — d=0 makes OpenCV infer a huge kernel)
    smooth = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    t_map = sauvola_threshold_map(smooth, window_size=45, k=0.34)
    ink = smooth.astype(np.float64) <= t_map

    binary = (ink.astype(np.uint8) * 255)
    k_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_el)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_el)

    cv2.imwrite(str(output_path), binary)


def binarize(
    image_path: str | Path,
    output_path: str | Path,
    *,
    prefer_nlbin: bool = True,
) -> None:
    """
    Binarize ``image_path`` and write ``output_path``.

    If ``prefer_nlbin`` is True, uses Kraken ``nlbin`` when available (same as API).
    Otherwise, or if Kraken/scipy is missing or fails, uses the local OpenCV fallback.
    """
    if prefer_nlbin:
        try:
            _binarize_nlbin(image_path, output_path)
            return
        except ImportError:
            warnings.warn(
                "Kraken not installed — using OpenCV fallback. "
                "Install the full backend stack (see requirements.txt) or `pip install kraken` "
                "to match `*_binarized.png` quality from the API.",
                stacklevel=2,
            )
        except Exception as e:
            warnings.warn(
                f"Kraken nlbin failed ({e!r}); falling back to OpenCV pipeline.",
                stacklevel=2,
            )

    _binarize_local_fallback(image_path, output_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Binarize manuscript images (Kraken nlbin when available, else OpenCV)."
    )
    p.add_argument(
        "input",
        nargs="?",
        default=str(_DEFAULT_INPUT),
        help="Input image path",
    )
    p.add_argument(
        "output",
        nargs="?",
        default=str(_REPO_ROOT / "binary.png"),
        help="Output PNG path",
    )
    p.add_argument(
        "--local",
        action="store_true",
        help="Force OpenCV fallback (skip Kraken even if installed)",
    )
    args = p.parse_args()
    binarize(args.input, args.output, prefer_nlbin=not args.local)
