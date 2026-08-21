from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping


_LANGUAGES = ("greek", "armenian", "syriac")
_SEQUENCE_LENGTH_METRIC = re.compile(
    r"sequence_length_(\d{3})_(\d{3})_(samples|reference_characters|character_errors|cer)$"
)


def _evaluation_scope_and_metric(key: str) -> tuple[str, str] | None:
    """Return the W&B section and flat metric suffix for evaluation keys."""
    if not key.startswith("eval_"):
        return None
    metric = key.removeprefix("eval_")
    for language in _LANGUAGES:
        prefix = f"{language}_"
        if metric.startswith(prefix):
            return language, metric.removeprefix(prefix)
    return "eval", metric


def _wandb_metric_key(key: str) -> str:
    """Map flat trainer metrics into W&B's slash-separated chart sections."""
    if key.startswith("train_"):
        return f"train/{key.removeprefix('train_')}"
    evaluation_metric = _evaluation_scope_and_metric(key)
    if evaluation_metric is not None:
        scope, metric = evaluation_metric
        return f"{scope}/{_metric_path(metric)}"
    if key in {"epoch", "grad_norm", "learning_rate"}:
        return f"train/{key}"
    return key


def _metric_path(metric: str) -> str:
    """Group sequence-length bin counters below their evaluation section."""
    match = _SEQUENCE_LENGTH_METRIC.fullmatch(metric)
    if match is None:
        return metric
    lower, upper, name = match.groups()
    return f"sequence_length/{lower}-{upper}/{name}"


def _sequence_length_rows(
    metrics: Mapping[str, float],
) -> tuple[dict[str, list[tuple[object, ...]]], set[str]]:
    """Collect scalar sequence-length metrics into per-evaluation bar-chart rows."""
    values: dict[str, dict[tuple[int, int], dict[str, float]]] = {}
    sequence_keys: set[str] = set()
    for key, value in metrics.items():
        evaluation_metric = _evaluation_scope_and_metric(str(key))
        if evaluation_metric is None:
            continue
        scope, metric = evaluation_metric
        match = _SEQUENCE_LENGTH_METRIC.fullmatch(metric)
        if match is None:
            continue
        lower, upper, name = match.groups()
        values.setdefault(scope, {}).setdefault(
            (int(lower), int(upper)), {}
        )[name] = float(value)
        sequence_keys.add(str(key))

    rows_by_scope: dict[str, list[tuple[object, ...]]] = {}
    for scope, bins in values.items():
        rows_by_scope[scope] = [
            (
                f"{lower}-{upper}",
                bin_metrics.get("cer", 0.0),
                bin_metrics.get("character_errors", 0.0),
                bin_metrics.get("reference_characters", 0.0),
                bin_metrics.get("samples", 0.0),
            )
            for (lower, upper), bin_metrics in sorted(bins.items())
        ]
    return rows_by_scope, sequence_keys


class WandbLogger:
    """Optional model-agnostic W&B run lifecycle and metric transport."""

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
            reinit=True,
        )

    @property
    def run(self) -> Any | None:
        """Expose run metadata needed by the training entry point."""
        return self._run

    def update_config(self, config: Mapping[str, Any]) -> None:
        """Record configuration values resolved after run initialization."""
        if self._run is not None:
            self._run.config.update(dict(config), allow_val_change=True)

    def log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None:
        """Log model metrics under W&B's training and evaluation chart sections."""
        if not self._run:
            return

        sequence_rows, sequence_keys = _sequence_length_rows(metrics)
        payload = {
            _wandb_metric_key(str(key)): float(value)
            for key, value in metrics.items()
            if str(key) not in sequence_keys
        }
        if sequence_rows:
            import wandb

            for scope, rows in sequence_rows.items():
                table = wandb.Table(
                    columns=[
                        "Reference character length",
                        "CER",
                        "Character errors",
                        "Reference characters",
                        "Samples",
                    ],
                    data=rows,
                )
                payload[f"{scope}/sequence_length_cer"] = wandb.plot.bar(
                    table,
                    "Reference character length",
                    "CER",
                    title=f"{scope.title()} CER by reference character length",
                )
        if not payload:
            return
        self._run.log(payload, step=step)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
