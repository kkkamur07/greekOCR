"""Training adapter for the original TrOCR image augmentations."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from .data_aug import DataAugment, OptForDataAugment


@dataclass
class LineAugmentation:
    """Apply the original TrOCR augmentation operators to one line image.

    The final resize, tensor conversion, and normalization remain the
    responsibility of the Hugging Face image processor. This adapter therefore
    invokes the original PIL-level random augmentation path only.
    """

    probability: float = 0.0
    mode: str = "random"
    num_operations: int = 3
    magnitude: int | None = None
    exclude_groups: tuple[str, ...] = ()
    exclude_operations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        mode_flags = {
            "isrand_aug": self.mode == "random",
            "issemantic_aug": self.mode == "semantic",
            "islearning_aug": self.mode == "learning",
            "isscatter_aug": self.mode == "scatter",
            "isrotation_aug": self.mode == "rotation",
        }
        if not any(mode_flags.values()):
            raise ValueError(
                "mode must be one of: random, semantic, learning, scatter, rotation"
            )
        self._augmenter = DataAugment(
            OptForDataAugment(
                eval=False,
                augs_num=self.num_operations,
                augs_mag=self.magnitude,
                issel_aug=False,
                **mode_flags,
            )
        )
        excluded_groups = set(self.exclude_groups)
        excluded_operations = set(self.exclude_operations)
        known_groups = {
            "process",
            "camera",
            "noise",
            "blur",
            "weather",
            "pattern",
            "warp",
            "geometry",
        }
        unknown_groups = excluded_groups - known_groups
        if unknown_groups:
            raise ValueError(f"Unknown augmentation groups: {sorted(unknown_groups)}")

        for group_name in known_groups:
            group = getattr(self._augmenter, group_name, None)
            if group is None:
                continue
            if group_name in excluded_groups:
                group.clear()
            else:
                group[:] = [
                    operation
                    for operation in group
                    if type(operation).__name__ not in excluded_operations
                ]
        self._augmenter.augs = [group for group in self._augmenter.augs if group]

    def __call__(self, image: Image.Image) -> Image.Image:
        from numpy.random import uniform

        if uniform(0, 1) >= self.probability:
            return image
        return self._augmenter.rand_aug(image.copy())
