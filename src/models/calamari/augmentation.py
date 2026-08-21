"""Random, print-aware line-image augmentation for the Calamari recognizer."""

from __future__ import annotations

import random
from collections.abc import Callable

import cv2
import numpy
from torch import Tensor


_OPERATIONS_PER_VARIANT = 3


def augment_legacy_line_image(image: Tensor) -> Tensor:
    """Apply three random Calamari transforms to a grayscale line image.

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
    augmented = _random_augment(ink)
    grayscale = numpy.clip((1.0 - augmented) * scale, 0.0, scale)
    return Tensor(grayscale.astype(pixels.dtype, copy=False)).to(dtype=original_dtype).unsqueeze(-1)


def _random_augment(image: numpy.ndarray) -> numpy.ndarray:
    """Select three distinct transformations from the Calamari augmentation pool."""
    operations: tuple[Callable[[numpy.ndarray], numpy.ndarray], ...] = (
        lambda value: _random_pad(value, (0, max(2, value.shape[1] * 2))),
        _random_rotate,
        _smooth_elastic_distortion,
        _printlike_degradation,
        _camera_degradation,
        _other_blur,
        _image_processing,
        _stroke_morphology,
    )
    augmented = image
    for operation in random.sample(operations, _OPERATIONS_PER_VARIANT):
        augmented = numpy.clip(operation(augmented), 0.0, 1.0)
    return augmented


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


def _random_rotate(image: numpy.ndarray, maximum_degrees: float = 5.0) -> numpy.ndarray:
    """Rotate a width-major line image by a uniformly sampled small angle."""
    width, height = image.shape[:2]
    angle = float(numpy.random.uniform(-maximum_degrees, maximum_degrees))
    transform = cv2.getRotationMatrix2D((height / 2, width / 2), angle, 1.0)
    return cv2.warpAffine(
        image,
        transform,
        (height, width),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _smooth_elastic_distortion(image: numpy.ndarray) -> numpy.ndarray:
    sigma = random.uniform(2.0, 5.0)
    return _distort_with_noise(image, _bounded_gaussian_noise(image.shape, sigma, 3.0))


def _printlike_degradation(image: numpy.ndarray) -> numpy.ndarray:
    return _printlike_multiscale(image, blur=1.0, inverted=True)


def _camera_degradation(image: numpy.ndarray) -> numpy.ndarray:
    return random.choice(
        (_adjust_brightness, _adjust_contrast, _jpeg_compression, _pixelate)
    )(image)


def _adjust_brightness(image: numpy.ndarray) -> numpy.ndarray:
    return image * random.uniform(0.75, 1.25)


def _adjust_contrast(image: numpy.ndarray) -> numpy.ndarray:
    factor = random.uniform(0.7, 1.3)
    return (image - image.mean()) * factor + image.mean()


def _jpeg_compression(image: numpy.ndarray) -> numpy.ndarray:
    encoded, buffer = cv2.imencode(
        ".jpg",
        numpy.rint(image * 255.0).astype(numpy.uint8),
        [cv2.IMWRITE_JPEG_QUALITY, random.randint(35, 75)],
    )
    if not encoded:
        return image
    decoded = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    return decoded.astype(numpy.float32) / 255.0


def _pixelate(image: numpy.ndarray) -> numpy.ndarray:
    width, height = image.shape
    scale = random.uniform(0.45, 0.75)
    reduced = cv2.resize(
        image,
        (max(1, round(height * scale)), max(1, round(width * scale))),
        interpolation=cv2.INTER_LINEAR,
    )
    return cv2.resize(reduced, (height, width), interpolation=cv2.INTER_NEAREST)


def _other_blur(image: numpy.ndarray) -> numpy.ndarray:
    if random.choice((True, False)):
        return cv2.blur(image, (random.choice((3, 5)),) * 2)
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
    return cv2.filter2D(image, -1, kernel)


def _image_processing(image: numpy.ndarray) -> numpy.ndarray:
    return random.choice(
        (_autocontrast, _equalize_histogram, _sharpen, _posterize)
    )(image)


def _autocontrast(image: numpy.ndarray) -> numpy.ndarray:
    minimum = float(image.min())
    maximum = float(image.max())
    if maximum == minimum:
        return image
    return (image - minimum) / (maximum - minimum)


def _equalize_histogram(image: numpy.ndarray) -> numpy.ndarray:
    equalized = cv2.equalizeHist(numpy.rint(image * 255.0).astype(numpy.uint8))
    return equalized.astype(numpy.float32) / 255.0


def _sharpen(image: numpy.ndarray) -> numpy.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0, borderType=cv2.BORDER_REFLECT)
    return image + random.uniform(0.2, 0.8) * (image - blurred)


def _posterize(image: numpy.ndarray) -> numpy.ndarray:
    levels = random.choice((16, 32, 64))
    return numpy.floor(image * (levels - 1)) / (levels - 1)


def _stroke_morphology(image: numpy.ndarray) -> numpy.ndarray:
    """Slightly thicken, thin, or punch holes in ink-bright text strokes."""
    operation = random.choice(("thicken", "thin", "holes"))
    if operation == "thicken":
        return cv2.dilate(
            image,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
    if operation == "thin":
        return cv2.erode(
            image,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
    transformed = image.copy()
    holes = (transformed > 0.5) & (numpy.random.random(transformed.shape) < 0.005)
    transformed[holes] = 0.0
    return transformed


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
