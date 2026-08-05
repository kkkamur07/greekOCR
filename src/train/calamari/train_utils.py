from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # hydra/omegaconf only exist in the training environment (they pull in tfaip and
    # TensorFlow). Keeping the import type-only leaves this module importable — and
    # therefore testable — from the repo venv.
    from omegaconf import DictConfig

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
REPO_ROOT = Path(__file__).resolve().parents[3]


def build_calamari_train_command() -> tuple[list[str], dict[str, str]]:
    """Return the interpreter invocation and env for the vendored Calamari trainer.

    Calamari is not installed as a package; it is imported from src/model/calamari via
    PYTHONPATH so the fork's preprocessing (notably the grayscale convention in
    calamari_ocr/utils/grayscale.py) is what actually trains.
    """
    calamari_root = REPO_ROOT / "src" / "model" / "calamari"
    if not (calamari_root / "calamari_ocr" / "scripts" / "train.py").is_file():
        raise FileNotFoundError(f"Local Calamari source not found at {calamari_root}")

    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    python_bin = str(venv_python) if venv_python.is_file() else sys.executable

    env = os.environ.copy()
    pythonpath_parts = [str(calamari_root)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    return [python_bin, "-m", "calamari_ocr.scripts.train"], env


def collect_images(split_dir: Path) -> list[str]:
    if not split_dir.is_dir():
        return []
    return [
        str(path.resolve())
        for path in sorted(split_dir.iterdir())
        if (path.is_file() or path.is_symlink())
        and not path.name.startswith(".")
        and path.suffix.lower() in IMAGE_EXTS
    ]


def validate_pack(pack_dir: Path) -> tuple[list[str], list[str]]:
    train_images = collect_images(pack_dir / "train")
    val_images = collect_images(pack_dir / "val")
    if not train_images or not val_images:
        raise FileNotFoundError(f"Expected train/ and val/ images under {pack_dir}")
    return train_images, val_images


def build_calamari_command(
    cfg: DictConfig,
    *,
    output_dir: Path,
    train_images: list[str],
    val_images: list[str],
    extra_args: Sequence[str] = (),
) -> tuple[list[str], dict[str, str]]:
    """Assemble the trainer argv shared by the train and finetune entrypoints.

    ``extra_args`` carries the flags only one entrypoint sets (warmstart, codec,
    learning rate). It is spliced in ahead of the image lists because those are
    variadic and have to stay last.
    """
    base_cmd, env = build_calamari_train_command()
    cmd = base_cmd + [
        "--network",
        str(cfg.model.network),
        "--n_augmentations",
        str(cfg.training.n_augmentations),
        "--trainer.output_dir",
        str(output_dir),
        "--trainer.epochs",
        str(cfg.training.epochs),
        "--early_stopping.n_to_go",
        str(cfg.training.early_stopping_patience),
        "--early_stopping.frequency",
        str(cfg.training.early_stopping_frequency),
        "--train.gt_extension",
        ".gt.txt",
        "--val.gt_extension",
        ".gt.txt",
    ]
    cmd.extend(extra_args)
    if uses_gpu(cfg):
        cmd.extend(["--device.gpus", str(cfg.training.gpu)])
    cmd.extend(["--train.images", *train_images, "--val.images", *val_images])
    return cmd, env


def uses_gpu(cfg: DictConfig) -> bool:
    """An unset or blank ``training.gpu`` means train on CPU."""
    return cfg.training.gpu is not None and str(cfg.training.gpu) != ""


def write_header(log_file: Path, title: str, rows: dict[str, object]) -> None:
    """Print the run banner to the terminal and start the log file with it."""
    lines = ["=" * 40, f"{title}: {datetime.now()}"]
    lines += [f"  {label + ':':<19}{value}" for label, value in rows.items()]
    lines.append("=" * 40)
    with log_file.open("w", encoding="utf-8") as handle:
        for line in lines:
            print(line)
            handle.write(line + "\n")


def stream_process(
    cmd: list[str],
    log_file: Path,
    cer_log_file: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as log_handle:
        cer_handle = cer_log_file.open("a", encoding="utf-8") if cer_log_file else None
        try:
            with subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            ) as proc:
                assert proc.stdout is not None
                for line in proc.stdout:
                    print(line, end="")
                    log_handle.write(line)
                    log_handle.flush()
                    if cer_handle and ("CER" in line or "val_CER" in line):
                        cer_handle.write(line)
                        cer_handle.flush()
            if proc.returncode != 0:
                raise SystemExit(proc.returncode)
        finally:
            if cer_handle:
                cer_handle.close()
