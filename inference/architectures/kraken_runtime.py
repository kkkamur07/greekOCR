"""Runtime compatibility for the inference-only Kraken surface."""

from __future__ import annotations

import sys
from functools import partial
from types import ModuleType


class _InferenceOnlyImageInputTransforms:
    """Minimal BLLA preprocessing replacement without training dataset imports."""

    def __init__(
        self,
        _batch: int,
        height: int,
        width: int,
        channels: int,
        pad: int | tuple[int, int] | tuple[int, int, int, int],
        *,
        valid_norm: bool = False,
        force_binarization: bool = False,
        dtype: object | None = None,
    ) -> None:
        if valid_norm or force_binarization:
            raise RuntimeError(
                "Kraken training transforms are unavailable in the inference-only runtime"
            )
        if channels not in {1, 3} or height <= 0:
            raise RuntimeError("Unsupported Kraken BLLA input shape")

        import torch
        from kraken.lib import functional_im_transforms as image_transforms
        from torchvision.transforms import v2

        mode_transform = (
            v2.Grayscale(num_output_channels=1)
            if channels == 1
            else partial(image_transforms.pil_to_mode, mode="RGB")
        )
        self.transforms: list[object] = [mode_transform]
        if width > 0:
            self.transforms.append(
                v2.Resize(
                    (height, width), interpolation=v2.InterpolationMode.LANCZOS, antialias=True
                )
            )
            pad = 0
        else:

            def resize_proportionally(image: object) -> object:
                image_height = getattr(image, "height")
                image_width = getattr(image, "width")
                return v2.functional.resize(
                    image,
                    [height, int(image_width * height / image_height)],
                    interpolation=v2.InterpolationMode.LANCZOS,
                    antialias=True,
                )

            self.transforms.append(resize_proportionally)
        if pad:
            self.transforms.append(v2.Pad(pad, fill=255))
        self.transforms.extend(
            [
                v2.PILToTensor(),
                v2.ToDtype(dtype or torch.float32, scale=True),
                image_transforms.tensor_invert,
                partial(image_transforms.tensor_permute, perm=(0, 1, 2)),
            ]
        )

    def __call__(self, image: object) -> object:
        for transform in self.transforms:
            image = transform(image)  # type: ignore[operator]
        return image


def prepare_kraken_inference_runtime() -> None:
    """Avoid loading Kraken's unused training dataset package.

    ``kraken.blla`` imports ``kraken.lib.dataset`` but never uses it for BLLA
    inference. That package imports PyArrow and training-only dataset classes,
    which add over 100 MiB to a frozen helper. Inference services never train
    or export models, so provide the import placeholder before loading BLLA.
    """

    dataset_module = ModuleType("kraken.lib.dataset")
    dataset_module.ImageInputTransforms = _InferenceOnlyImageInputTransforms
    sys.modules.setdefault("kraken.lib.dataset", dataset_module)
