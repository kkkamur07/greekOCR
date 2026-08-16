"""Canonical line-image grayscale load, shared with the serving preprocessing.

This module deliberately imports nothing beyond PIL and numpy.  The rest of the
vendored Calamari tree needs ``paiargparse``/``tfaip`` (installed only in the
training environment), and the train/serve parity test has to be able to import
this conversion without them.
"""

import numpy as np
from PIL import Image


def load_line_image_grayscale(image_path) -> np.ndarray:
    """Load a line image as uint8 grayscale the way production serves it.

    ``inference/architectures/calamari/preprocessing/pipeline.py`` reads request
    bytes with ``Image.convert("L")``, so training has to use the same call or
    the model is served a different pixel distribution than it was fitted on.
    The two conversions are not interchangeable:

    * For palette ("P"), CMYK and 16-bit sources, reading the raw array and
      running ``cv.cvtColor`` operates on palette indices / ink values rather
      than colour, and the results differ by ~100 grey levels on average.
      ``convert("L")`` expands the palette and converts the colour space first.
    * "LA" (gray + alpha) has two channels, which the raw-array path rejects
      outright.
    * Even for plain 8-bit RGB the two differ by one level on ~0.1% of pixels;
      the luminance coefficients are identical (ITU-R 601-2) but OpenCV rounds
      where PIL truncates.

    ``convert("L")`` handles every PIL mode, so no channel dispatch is needed.
    """
    with Image.open(image_path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)
