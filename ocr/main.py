"""
Run Calamari training on ``<repo>/dataset/calamari/{train,val}/`` (line images + ``*.gt.txt``).

Calamari writes automatically to output_dir:
  train.log      — full training log with params and per-epoch metrics
  tensorboard/   — TensorBoard event files (run: tensorboard --logdir <output_dir>)
  best/          — continuously updated best model (lowest val loss)
  checkpoint*/   — checkpoint saved every epoch (used for --resume)

Optional env:
  CALAMARI_OUTPUT  — output dir (default ``outputs/calamari-greek-bible``)
  CALAMARI_GPU     — GPU id (default ``0``; set to "" to use CPU)

Usage:
  python -m ocr.main
  python -m ocr.main --epochs 50 --gpu 1
  python -m ocr.main --early-stopping-patience 15
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

DEFAULT_PACK = _REPO / "dataset" / "calamari"
DEFAULT_OUTPUT = _REPO / "outputs" / "calamari-greek-bible"

IMAGE_EXT = {".png", ".jpg", ".jpeg"}

log = logging.getLogger(__name__)


def _setup_logging(output_dir: Path) -> Path:
    """Configure root logger to write to console + a timestamped log file."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{timestamp}.log"

    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    return log_file


def _gpu_info() -> str:
    """Return nvidia-smi summary, or a fallback message if not available."""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        return result.stdout.strip()
    except Exception:
        return "(nvidia-smi not available)"


def _list_line_images(split_dir: Path) -> list[str]:
    if not split_dir.is_dir():
        return []
    return [
        str(p.absolute())
        for p in sorted(split_dir.iterdir())
        if (p.is_file() or p.is_symlink()) and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXT
    ]


def train(
    pack_dir: Path,
    output_dir: Path,
    *,
    gpu_ids: list[str],
    epochs: int,
    n_augmentations: int,
    network: str,
    early_stopping_patience: int,
    early_stopping_freq: int,
) -> None:
    train_imgs = _list_line_images(pack_dir / "train")
    val_imgs = _list_line_images(pack_dir / "val")
    if not train_imgs:
        raise SystemExit(f"No images under {pack_dir / 'train'} (*.png / *.jpg next to *.gt.txt).")
    if not val_imgs:
        raise SystemExit(f"No images under {pack_dir / 'val'}.")

    log.info("=" * 60)
    log.info("Calamari training starting")
    log.info("  Pack dir:          %s", pack_dir)
    log.info("  Output dir:        %s", output_dir)
    log.info("  Train images:      %d", len(train_imgs))
    log.info("  Val images:        %d", len(val_imgs))
    log.info("  Network:           %s", network)
    log.info("  Epochs:            %d", epochs)
    log.info("  Augmentations:     %d", n_augmentations)
    log.info("  Early stopping:    patience=%d epochs, check every %d epoch(s)", early_stopping_patience, early_stopping_freq)
    log.info("  (Calamari saves checkpoints every epoch and best/ model automatically)")
    log.info("  GPUs:              %s", gpu_ids if gpu_ids else "CPU")
    log.info("=" * 60)
    log.info("GPU info:\n%s", _gpu_info())
    log.info("=" * 60)

    cmd: list[str] = [
        "calamari-train",
        "--network", network,
        "--n_augmentations", str(n_augmentations),
        "--trainer.output_dir", str(output_dir),
        "--trainer.epochs", str(epochs),
        "--early_stopping.n_to_go", str(early_stopping_patience),
        "--early_stopping.frequency", str(early_stopping_freq),
        "--train.gt_extension", ".gt.txt",
        "--val.gt_extension", ".gt.txt",
        "--train.images", *train_imgs,
        "--val.images", *val_imgs,
    ]
    if gpu_ids:
        cmd.extend(["--device.gpus", *gpu_ids])

    log.info("Running: %s", " ".join(cmd))

    # Stream subprocess output line-by-line so it appears live in both
    # the console and the log file (via the handlers set up in _setup_logging).
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            log.info(line.rstrip())

    if proc.returncode != 0:
        raise SystemExit(f"calamari-train exited with code {proc.returncode}")

    log.info("=" * 60)
    log.info("Training complete. Checkpoints saved under: %s", output_dir)


def main(argv: list[str] | None = None) -> None:
    out_default = Path(os.environ.get("CALAMARI_OUTPUT", str(DEFAULT_OUTPUT))).expanduser().resolve()

    p = argparse.ArgumentParser(description="Calamari train on dataset/calamari.")
    p.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK,
                   help=f"Pack with train/ and val/ (default: {DEFAULT_PACK}).")
    p.add_argument("--output-dir", type=Path, default=out_default,
                   help=f"Checkpoints and logs (default: $CALAMARI_OUTPUT or {out_default}).")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--n-augmentations", type=int, default=5)
    p.add_argument("--network", default="def")
    p.add_argument("--gpu", default=None,
                   help="GPU id(s), e.g. '0' or '0 1'. Default: $CALAMARI_GPU or 0. Pass '' for CPU.")
    p.add_argument("--early-stopping-patience", type=int, default=20,
                   help="Stop if val loss does not improve for this many epochs (default: 20; -1 = disabled).")
    p.add_argument("--early-stopping-freq", type=int, default=1,
                   help="Check early stopping every N epochs (default: 1).")

    args = p.parse_args(argv)

    pack_dir = args.pack_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = _setup_logging(output_dir)

    if args.gpu is not None:
        g = args.gpu.strip()
        gpu_ids = g.split() if g else []
    else:
        raw = os.environ.get("CALAMARI_GPU", "0").strip()
        gpu_ids = raw.split() if raw else []

    log.info("Log file: %s", log_file)

    train(
        pack_dir,
        output_dir,
        gpu_ids=gpu_ids,
        epochs=args.epochs,
        n_augmentations=args.n_augmentations,
        network=args.network,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_freq=args.early_stopping_freq,
    )


if __name__ == "__main__":
    main()
