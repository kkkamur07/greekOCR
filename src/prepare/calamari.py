"""Create a canonical Calamari training pack through the Hydra configuration."""

from __future__ import annotations

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from ..models.calamari.preprocessing.preprocess import (
    prepare_pack,
    prepare_trocr_manifest_pack,
)


@hydra.main(version_base=None, config_path="../../config/calamari", config_name="configs")
def main(cfg: DictConfig) -> None:
    arguments = (
        to_absolute_path(cfg.preparation.raw_root),
        to_absolute_path(cfg.preparation.output_dir),
    )
    kwargs = {
        "copy_images": bool(cfg.preparation.copy_images),
        "force": bool(cfg.preparation.force),
    }
    source_format = str(cfg.preparation.source_format)
    if source_format == "trocr_manifest":
        total = prepare_trocr_manifest_pack(*arguments, **kwargs)
    elif source_format == "line_crops":
        total = prepare_pack(*arguments, **kwargs)
    else:
        raise ValueError(
            "preparation.source_format must be 'trocr_manifest' or 'line_crops'; "
            f"received {source_format!r}."
        )
    print(f"Prepared {total} Calamari line images.")


if __name__ == "__main__":
    main()
