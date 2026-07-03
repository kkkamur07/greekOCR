from __future__ import annotations

from pathlib import Path
from typing import Any


class WandbLogger:
    """Optional W&B logger for Calamari metric rows."""

    def __init__(
        self,
        *,
        enabled: bool,
        project: str,
        entity: str | None,
        name: str | None,
        mode: str,
        save_dir: Path,
        config: dict[str, Any],
    ) -> None:
        self.enabled = enabled
        self._run = None
        if not enabled:
            return

        import wandb

        self._run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            mode=mode,
            dir=str(save_dir),
            config=config,
            sync_tensorboard=True,
            reinit=True,
        )

    def log_row(self, row: dict[str, object]) -> None:
        if not self._run:
            return

        import wandb

        stage = str(row["stage"])
        step = int(row["epoch"])
        payload = {
            f"{stage}/{key}": float(value)
            for key, value in row.items()
            if key not in {"stage", "epoch"} and value != ""
        }
        if not payload:
            return
        payload["epoch"] = step
        wandb.log(payload, step=step)

    def finish(self) -> None:
        if self._run:
            self._run.finish()
