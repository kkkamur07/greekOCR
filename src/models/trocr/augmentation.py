"""Compact, dependency-light image augmentation for TrOCR line recognition."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
import random
from typing import Callable

import cv2
import numpy
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


ImageTransform = Callable[[Image.Image], Image.Image]


@dataclass
class LineAugmentation:
    """Apply exactly ``num_operations`` random transforms to a line image."""

    probability: float = 1.0
    num_operations: int = 3
    max_rotation_degrees: float = 5.0
    _operations: tuple[ImageTransform, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("TrOCR augmentation probability must be between zero and one.")
        if not 0.0 < self.max_rotation_degrees <= 5.0:
            raise ValueError("TrOCR rotation must be greater than zero and at most five degrees.")
        self._operations = (
            _gaussian_blur,
            lambda image: _random_rotate(image, self.max_rotation_degrees),
            _smooth_elastic_distortion,
            _camera_degradation,
            _other_blur,
            _image_processing,
            _stroke_morphology,
        )
        if not 1 <= self.num_operations <= len(self._operations):
            raise ValueError(
                f"TrOCR num_operations must be between one and {len(self._operations)}."
            )

    def __call__(self, image: Image.Image) -> Image.Image:
        """Return an RGB line image with the configured random transforms."""
        if random.random() >= self.probability:
            return image
        augmented = image.convert("RGB")
        for operation in random.sample(self._operations, self.num_operations):
            augmented = operation(augmented)
        return augmented


def _gaussian_blur(image: Image.Image) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))


def _random_rotate(image: Image.Image, maximum_degrees: float) -> Image.Image:
    angle = random.uniform(-maximum_degrees, maximum_degrees)
    return image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor="white")


def _smooth_elastic_distortion(image: Image.Image) -> Image.Image:
    pixels = numpy.asarray(image)
    height, width = pixels.shape[:2]
    delta_x = _smooth_displacement((height, width), sigma=random.uniform(4.0, 8.0))
    delta_y = _smooth_displacement((height, width), sigma=random.uniform(4.0, 8.0))
    coordinates_x, coordinates_y = numpy.meshgrid(
        numpy.arange(width, dtype=numpy.float32),
        numpy.arange(height, dtype=numpy.float32),
    )
    distorted = cv2.remap(
        pixels,
        coordinates_x + delta_x,
        coordinates_y + delta_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return Image.fromarray(distorted, mode="RGB")


def _smooth_displacement(shape: tuple[int, int], *, sigma: float, maximum_offset: float = 3.0) -> numpy.ndarray:
    values = numpy.random.uniform(-1.0, 1.0, shape).astype(numpy.float32)
    smoothed = cv2.GaussianBlur(values, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT)
    maximum = float(numpy.abs(smoothed).max())
    return smoothed if maximum == 0 else smoothed * (maximum_offset / maximum)


def _camera_degradation(image: Image.Image) -> Image.Image:
    operation = random.choice(
        (_adjust_brightness, _adjust_contrast, _jpeg_compression, _pixelate)
    )
    return operation(image)


def _adjust_brightness(image: Image.Image) -> Image.Image:
    return ImageEnhance.Brightness(image).enhance(random.uniform(0.7, 1.3))


def _adjust_contrast(image: Image.Image) -> Image.Image:
    return ImageEnhance.Contrast(image).enhance(random.uniform(0.7, 1.3))


def _jpeg_compression(image: Image.Image) -> Image.Image:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=random.randint(35, 75))
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB").copy()


def _pixelate(image: Image.Image) -> Image.Image:
    scale = random.uniform(0.45, 0.75)
    downsampled = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.BILINEAR,
    )
    return downsampled.resize(image.size, Image.Resampling.NEAREST)


def _other_blur(image: Image.Image) -> Image.Image:
    pixels = numpy.asarray(image)
    if random.choice((True, False)):
        blurred = cv2.blur(pixels, (random.choice((3, 5)),) * 2)
    else:
        kernel_size = random.choice((5, 7, 9))
        kernel = numpy.zeros((kernel_size, kernel_size), dtype=numpy.float32)
        cv2.line(
            kernel,
            (0, kernel_size // 2),
            (kernel_size - 1, kernel_size // 2),
            1.0,
            1,
        )
        kernel /= kernel.sum()
        blurred = cv2.filter2D(pixels, -1, kernel)
    return Image.fromarray(blurred, mode="RGB")


def _image_processing(image: Image.Image) -> Image.Image:
    operation = random.choice((_autocontrast, ImageOps.equalize, _sharpen, _posterize))
    return operation(image)


def _autocontrast(image: Image.Image) -> Image.Image:
    return ImageOps.autocontrast(image, cutoff=random.choice((0, 1, 2)))


def _sharpen(image: Image.Image) -> Image.Image:
    return ImageEnhance.Sharpness(image).enhance(random.uniform(1.2, 2.0))


def _posterize(image: Image.Image) -> Image.Image:
    return ImageOps.posterize(image, bits=random.choice((4, 5, 6)))


def _stroke_morphology(image: Image.Image) -> Image.Image:
    """Slightly thicken, thin, or punch holes in dark text strokes."""
    pixels = numpy.asarray(image)
    operation = random.choice(("thicken", "thin", "holes"))
    if operation == "thicken":
        transformed = cv2.erode(
            pixels,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
    elif operation == "thin":
        transformed = cv2.dilate(
            pixels,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
    else:
        transformed = pixels.copy()
        darkness = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY) < 128
        holes = darkness & (numpy.random.random(darkness.shape) < 0.005)
        transformed[holes] = 255
    return Image.fromarray(transformed, mode="RGB")
