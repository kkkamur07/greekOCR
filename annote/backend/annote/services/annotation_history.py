"""Annotation history — timed and milestone snapshots with retention."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException

from annote.schemas.annotation import PageAnnotation
from annote.schemas.history import HistoryListResponse, HistorySnapshotRecord, HistorySnapshotSummary
from annote.services.annotation_store import load_annotation, save_annotation
from annote.services.page_lock import assert_page_unlocked
from annote.services.segment_text import compute_pairing_progress
from annote.services.text_lines import split_text_lines
from annote.settings import HistorySettings, get_settings


def _history_dir(data_root: Path, stem: str) -> Path:
    return data_root / "annotations" / "history" / stem


def _state_path(data_root: Path, stem: str) -> Path:
    return _history_dir(data_root, stem) / "_state.json"


def _load_state(data_root: Path, stem: str) -> dict:
    path = _state_path(data_root, stem)
    if not path.is_file():
        return {"last_timed_snapshot_at": None, "captured_milestones": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(data_root: Path, stem: str, state: dict) -> None:
    path = _state_path(data_root, stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _pairing_percent(data_root: Path, stem: str, annotation: PageAnnotation) -> int:
    transcription_path = data_root / "transcriptions" / "pages" / f"{stem}.txt"
    raw_text = transcription_path.read_text(encoding="utf-8") if transcription_path.is_file() else ""
    text_lines = split_text_lines(raw_text) if raw_text.strip() else []
    progress = compute_pairing_progress(annotation.segments, text_lines)
    if not annotation.segments:
        return 0
    return round(100 * progress.paired_count / len(annotation.segments))


def _list_records(data_root: Path, stem: str) -> list[HistorySnapshotRecord]:
    history_dir = _history_dir(data_root, stem)
    if not history_dir.is_dir():
        return []
    records: list[HistorySnapshotRecord] = []
    for path in sorted(history_dir.glob("*.json")):
        if path.name == "_state.json":
            continue
        records.append(HistorySnapshotRecord.model_validate_json(path.read_text(encoding="utf-8")))
    records.sort(key=lambda r: r.timestamp)
    return records


def list_history(data_root: Path, stem: str) -> HistoryListResponse:
    snapshots = [
        HistorySnapshotSummary(
            id=r.id,
            timestamp=r.timestamp,
            reason=r.reason,
            pairing_progress_percent=r.pairing_progress_percent,
        )
        for r in _list_records(data_root, stem)
    ]
    return HistoryListResponse(snapshots=snapshots)


def _write_snapshot(
    data_root: Path,
    stem: str,
    annotation: PageAnnotation,
    *,
    reason: str,
    protected: bool,
) -> HistorySnapshotRecord:
    history_dir = _history_dir(data_root, stem)
    history_dir.mkdir(parents=True, exist_ok=True)
    record = HistorySnapshotRecord(
        id=uuid.uuid4().hex[:12],
        timestamp=datetime.now(timezone.utc).isoformat(),
        reason=reason,
        pairing_progress_percent=_pairing_percent(data_root, stem, annotation),
        protected=protected,
        annotation=annotation.model_copy(deep=True),
    )
    path = history_dir / f"{record.id}.json"
    path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    if not protected:
        _prune_timed_snapshots(data_root, stem)
    return record


def _prune_timed_snapshots(data_root: Path, stem: str) -> None:
    settings = get_settings().history
    records = [r for r in _list_records(data_root, stem) if not r.protected]
    excess = len(records) - settings.max_timed_snapshots
    if excess <= 0:
        return
    history_dir = _history_dir(data_root, stem)
    for record in records[:excess]:
        path = history_dir / f"{record.id}.json"
        if path.is_file():
            path.unlink()


def capture_snapshot(
    data_root: Path,
    stem: str,
    annotation: PageAnnotation,
    *,
    reason: str,
    protected: bool,
) -> HistorySnapshotRecord:
    return _write_snapshot(data_root, stem, annotation, reason=reason, protected=protected)


def maybe_capture_on_save(data_root: Path, stem: str, annotation: PageAnnotation) -> None:
    settings = get_settings().history
    state = _load_state(data_root, stem)
    now = datetime.now(timezone.utc)
    percent = _pairing_percent(data_root, stem, annotation)

    for milestone in settings.pairing_milestones:
        key = f"milestone_{milestone}"
        if percent >= milestone and milestone not in state.get("captured_milestones", []):
            capture_snapshot(data_root, stem, annotation, reason=key, protected=True)
            captured = list(state.get("captured_milestones", []))
            captured.append(milestone)
            state["captured_milestones"] = sorted(set(captured))

    last_at = state.get("last_timed_snapshot_at")
    interval_seconds = settings.snapshot_interval_minutes * 60
    if last_at is None or (now - datetime.fromisoformat(last_at)).total_seconds() >= interval_seconds:
        capture_snapshot(data_root, stem, annotation, reason="timed", protected=False)
        state["last_timed_snapshot_at"] = now.isoformat()

    _save_state(data_root, stem, state)


def restore_snapshot(data_root: Path, stem: str, snapshot_id: str) -> PageAnnotation:
    current = load_annotation(data_root, stem)
    assert_page_unlocked(current)

    path = _history_dir(data_root, stem) / f"{snapshot_id}.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"History snapshot not found: {snapshot_id}")

    record = HistorySnapshotRecord.model_validate_json(path.read_text(encoding="utf-8"))
    restored = record.annotation.model_copy(deep=True)
    restored.locked = current.locked
    restored.export_metadata = None
    return save_annotation(data_root, stem, restored)
