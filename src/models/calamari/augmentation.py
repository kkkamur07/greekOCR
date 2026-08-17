"""Legacy Calamari line-image augmentation implemented for PyTorch datasets."""

from __future__ import annotations

import random

import cv2
import numpy
from torch import Tensor


def augment_legacy_line_image(image: Tensor) -> Tensor:
    """Apply Calamari's original padding, distortion, and print-like degradation.

    Calamari's historical augmenter operated on ink-bright, zero-background
    arrays. PyTorch Calamari stores grayscale image intensities instead, so the
    image is inverted for the transformation and inverted again before return.
    """
    if image.ndim != 3 or image.shape[-1] != 1:
        raise ValueError("Calamari augmentation requires a (width, height, 1) image tensor.")

    original_dtype = image.dtype
    pixels = image.squeeze(-1).detach().cpu().numpy().astype(numpy.float32, copy=False)
    scale = 255.0 if float(pixels.max(initial=0.0)) > 1.0 else 1.0
    ink = 1.0 - numpy.clip(pixels / scale, 0.0, 1.0)
    augmented = _legacy_augment(ink)
    grayscale = numpy.clip((1.0 - augmented) * scale, 0.0, scale)
    return Tensor(grayscale.astype(pixels.dtype, copy=False)).to(dtype=original_dtype).unsqueeze(-1)


def _legacy_augment(image: numpy.ndarray) -> numpy.ndarray:
    padded = _random_pad(image, (0, max(2, image.shape[1] * 2)))
    for sigma in (2, 5):
        padded = _distort_with_noise(padded, _bounded_gaussian_noise(padded.shape, sigma, 3.0))
    printed = _printlike_multiscale(padded, blur=1.0, inverted=True)
    maximum = float(printed.max(initial=0.0))
    return printed if maximum == 0 else printed / maximum


def _random_pad(image: numpy.ndarray, horizontal: tuple[int, int]) -> numpy.ndarray:
    left, right = numpy.random.randint(*horizontal, size=2)
    return cv2.copyMakeBorder(
        image,
        int(left),
        int(right),
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=0,
    )


def _bounded_gaussian_noise(shape: tuple[int, ...], sigma: float, maxdelta: float) -> numpy.ndarray:
    width, height = shape[:2]
    deltas = numpy.random.rand(2, width, height)
    for axis, values in enumerate(deltas):
        deltas[axis] = cv2.GaussianBlur(
            values, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT
        )
    deltas -= deltas.min()
    deltas /= deltas.max()
    return (2 * deltas - 1) * maxdelta


def _distort_with_noise(image: numpy.ndarray, deltas: numpy.ndarray) -> numpy.ndarray:
    if deltas.shape != (2, *image.shape[:2]):
        raise ValueError("Calamari distortion offsets must match image dimensions.")
    width, height = image.shape[:2]
    coordinates = numpy.transpose(
        numpy.array(numpy.meshgrid(range(width), range(height))),
        axes=[0, 2, 1],
    )
    deltas = deltas + coordinates
    return cv2.remap(
        image,
        deltas[1].astype(numpy.float32),
        deltas[0].astype(numpy.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def _random_blotches(image: numpy.ndarray, foreground: float, background: float) -> numpy.ndarray:
    fg = _random_blobs(image.shape[:2], foreground, 10)
    bg = _random_blobs(image.shape[:2], background, 10)
    return numpy.minimum(numpy.maximum(image, fg), 1 - bg)


def _random_blobs(shape: tuple[int, int], density: float, size: int) -> numpy.ndarray:
    width, height = shape
    count = max(1, int(density * width * height))
    mask = numpy.zeros((width, height), numpy.uint8)
    for _ in range(count):
        mask[random.randint(0, width - 1), random.randint(0, height - 1)] = 1
    distance = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3)
    mask = numpy.array(distance < size, dtype=numpy.float32)
    mask = cv2.GaussianBlur(
        mask, (0, 0), sigmaX=size / 4, borderType=cv2.BORDER_REFLECT
    )
    mask -= mask.min()
    maximum = float(mask.max())
    if maximum:
        mask /= maximum
    noise = cv2.GaussianBlur(
        numpy.random.rand(width, height),
        (0, 0),
        sigmaX=size / 4,
        borderType=cv2.BORDER_REFLECT,
    )
    noise -= noise.min()
    maximum = float(noise.max())
    if maximum:
        noise /= maximum
    return numpy.array(mask * noise > 0.5, dtype=numpy.float32)


def _make_multiscale_noise_uniform(
    shape: tuple[int, int], *, scale_range: tuple[float, float] = (1.0, 100.0)
) -> numpy.ndarray:
    minimum, maximum = numpy.log10(scale_range)
    scales = numpy.random.uniform(size=4)
    scales = numpy.add.accumulate(scales)
    scales -= scales.min()
    scales /= scales.max()
    scales = 10 ** (scales * (maximum - minimum) + minimum)
    weights = 2.0 * numpy.random.uniform(size=4)
    result = _make_noise_at_scale(shape, scales[0]) * weights[0]
    for scale, weight in zip(scales, weights, strict=True):
        result += _make_noise_at_scale(shape, scale) * weight
    result -= result.min()
    result /= result.max()
    return result


def _make_noise_at_scale(shape: tuple[int, int], scale: float) -> numpy.ndarray:
    width, height = shape
    source = numpy.random.rand(int(width / scale + 1), int(height / scale + 1))
    noise = cv2.resize(source, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return noise[:width, :height]


def _printlike_multiscale(
    image: numpy.ndarray,
    *,
    blur: float,
    inverted: bool,
) -> numpy.ndarray:
    selector = image if inverted else 1 - image
    selector = _random_blotches(selector, 3 * 5e-5, 5e-5)
    paper = 0.8 + 0.2 * _make_multiscale_noise_uniform(image.shape[:2])
    ink = 0.2 * _make_multiscale_noise_uniform(image.shape[:2])
    blurred = (
        cv2.GaussianBlur(selector, (0, 0), sigmaX=blur, borderType=cv2.BORDER_REFLECT)
        + selector
    ) / 2
    printed = blurred * ink + (1 - blurred) * paper
    return 1 - printed if inverted else printed
