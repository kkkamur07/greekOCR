from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
REPO_ROOT = Path(__file__).resolve().parents[3]


def locate_python_bin() -> str:
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    if venv_python.is_file():
        return str(venv_python)
    return sys.executable


def locate_local_calamari_root() -> Path:
    calamari_root = REPO_ROOT / "src" / "model" / "calamari"
    expected_train_module = calamari_root / "calamari_ocr" / "scripts" / "train.py"
    if not expected_train_module.is_file():
        raise FileNotFoundError(f"Local Calamari source not found at {calamari_root}")
    return calamari_root


def build_calamari_train_command() -> tuple[list[str], dict[str, str]]:
    calamari_root = locate_local_calamari_root()
    env = os.environ.copy()
    pythonpath_parts = [str(calamari_root)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    return [locate_python_bin(), "-m", "calamari_ocr.scripts.train"], env


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


def load_scalars(directory: Path, tag: str) -> dict[int, float]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"Could not import TensorBoard event reader: {exc}")
        return {}

    if not directory.is_dir():
        return {}

    try:
        accumulator = EventAccumulator(str(directory))
        accumulator.Reload()
        if tag not in accumulator.Tags().get("scalars", []):
            return {}
        return {event.step: event.value for event in accumulator.Scalars(tag)}
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"Could not read metrics from {directory}: {exc}")
        return {}


def write_metrics_csv(output_dir: Path, metrics_file: Path) -> None:
    rows: list[dict[str, object]] = []
    for stage, stage_dir in (("main", output_dir), ("aug_data", output_dir / "aug_data"), ("real_data", output_dir / "real_data")):
        train_dir = stage_dir / "train"
        val_dir = stage_dir / "validation"
        train_cer = load_scalars(train_dir, "epoch_CER")
        train_loss = load_scalars(train_dir, "epoch_ctc-loss")
        val_cer = load_scalars(val_dir, "epoch_CER")
        val_loss = load_scalars(val_dir, "epoch_ctc-loss")
        for step in sorted(set(train_cer) | set(train_loss) | set(val_cer) | set(val_loss)):
            rows.append(
                {
                    "stage": stage,
                    "epoch": step,
                    "train_cer": train_cer.get(step, ""),
                    "train_ctc_loss": train_loss.get(step, ""),
                    "val_cer": val_cer.get(step, ""),
                    "val_ctc_loss": val_loss.get(step, ""),
                }
            )

    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with metrics_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["stage", "epoch", "train_cer", "train_ctc_loss", "val_cer", "val_ctc_loss"],
        )
        writer.writeheader()
        writer.writerows(rows)
