#!/usr/bin/env python3
"""Import legacy on-disk annotated corpus into the dev@example.com platform user.

Reads the Option D layout under ``data/annotated/data/`` (or ``ANNOTATED_DATA_ROOT``):

  manuscripts/pages/        — page images
  annotations/pages/        — current segment geometry + pairings
  annotations/history/      — restorable annotation snapshots (per page stem)
  transcriptions/pages/     — optional page transcription text (line-broken)

Creates one owned project with documents grouped by manuscript prefix
(e.g. ``Grec_1360_..._6/7/8`` become one multi-part document). Images are copied
into ``MEDIA_ROOT``; structured data is written to Postgres.

Idempotent: skips documents that already exist in the target project (by name).
When a document is skipped, missing history snapshots are backfilled automatically.

Usage (from annote/):

  PYTHONPATH=. python scripts/seed_dev_annotated_data.py
  PYTHONPATH=. python scripts/seed_dev_annotated_data.py --force
  PYTHONPATH=. python scripts/seed_dev_annotated_data.py --skip-history
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import UUID

# Quiet SQL echo unless --verbose (engine is created at import time).
if "--verbose" not in sys.argv and "-v" not in sys.argv:
    os.environ.setdefault("ENVIRONMENT", "production")

from PIL import Image
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.annotation.infrastructure.orm_models import AnnotationHistorySnapshot
from backend.core.settings._env import REPO_ROOT
from backend.document.application.document_service import DocumentService
from backend.document.infrastructure.media_store import MediaStore
from backend.document.infrastructure.orm_models import (
    Document,
    DocumentPart,
    LineGeometryKind,
    LineSource,
)
from backend.project.application.project_service import ProjectService
from backend.project.infrastructure.orm_models import Project
from backend.users.application.auth_service import AuthService
from backend.users.infrastructure.orm_models import User
from infrastructure import models as _orm_models  # noqa: F401 — register all mappers
from infrastructure.db import AsyncSessionLocal

log = logging.getLogger("seed_dev_annotated_data")

DEV_EMAIL = os.environ.get("DEV_USER_EMAIL", "dev@example.com")
DEV_USERNAME = os.environ.get("DEV_USER_USERNAME", "dev")
DEV_PASSWORD = os.environ.get("DEV_USER_PASSWORD", "dev-pass-123")

PROJECT_SLUG = os.environ.get("DEV_ANNOTATED_PROJECT_SLUG", "dev-annotated-corpus")
PROJECT_NAME = os.environ.get("DEV_ANNOTATED_PROJECT_NAME", "Dev annotated corpus")
PROJECT_GUIDELINES = os.environ.get(
    "DEV_ANNOTATED_PROJECT_GUIDELINES",
    "Sample manuscript pages imported from data/annotated/data for local development.",
)

DEFAULT_DATA_ROOT = REPO_ROOT.parent / "data" / "annotated" / "data"
ANNOTATED_DATA_ROOT = Path(os.environ.get("ANNOTATED_DATA_ROOT", str(DEFAULT_DATA_ROOT))).resolve()

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")
_PAGE_SUFFIX_RE = re.compile(r"^(.+)_(\d+)$")
_IMAGE_KEY_STEM_RE = re.compile(r"/([^/]+)\.[a-z0-9]{1,16}$")


@dataclass
class ImportStats:
    documents_created: int = 0
    documents_skipped: int = 0
    parts: int = 0
    segments: int = 0
    history_snapshots: int = 0
    warnings: list[str] = field(default_factory=list)

    def merge(self, other: ImportStats) -> None:
        self.documents_created += other.documents_created
        self.documents_skipped += other.documents_skipped
        self.parts += other.parts
        self.segments += other.segments
        self.history_snapshots += other.history_snapshots
        self.warnings.extend(other.warnings)


def _configure_logging(*, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")
    logging.getLogger("sqlalchemy.engine").setLevel(logging.DEBUG if verbose else logging.ERROR)


def _page_sort_key(stem: str) -> tuple[str, int]:
    match = _PAGE_SUFFIX_RE.match(stem)
    if match is None:
        return (stem, 0)
    return (match.group(1), int(match.group(2)))


def _group_pages_by_document(page_stems: list[str]) -> dict[str, list[str]]:
    """Group page stems into documents; multi-page manuscripts share a prefix."""
    prefix_counts: dict[str, int] = {}
    for stem in page_stems:
        match = _PAGE_SUFFIX_RE.match(stem)
        if match is not None:
            prefix_counts[match.group(1)] = prefix_counts.get(match.group(1), 0) + 1

    groups: dict[str, list[str]] = {}
    for stem in sorted(page_stems):
        match = _PAGE_SUFFIX_RE.match(stem)
        if match is not None and prefix_counts[match.group(1)] > 1:
            groups.setdefault(match.group(1), []).append(stem)
        else:
            groups[stem] = [stem]

    for doc_key, stems in groups.items():
        if len(stems) > 1:
            groups[doc_key] = sorted(stems, key=_page_sort_key)
    return groups


def _document_display_name(doc_key: str, page_stems: list[str]) -> str:
    if len(page_stems) == 1:
        return page_stems[0]
    if doc_key.startswith("Grec_1360_"):
        return "Grec 1360 — Harmenopulus (btv1b10721710m)"
    return doc_key.replace("_", " ")


def _stem_from_image_key(image_key: str) -> str | None:
    match = _IMAGE_KEY_STEM_RE.search(image_key)
    return match.group(1) if match else None


def _resolve_page_image(pages_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = pages_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_annotation(annotations_dir: Path, stem: str) -> dict[str, Any] | None:
    path = annotations_dir / f"{stem}.json"
    return _load_json(path) if path.is_file() else None


def _load_page_transcription(transcriptions_dir: Path, stem: str) -> str | None:
    path = transcriptions_dir / f"{stem}.txt"
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    return text if text.strip() else None


def _approved_text(segment: dict[str, Any]) -> str | None:
    text = segment.get("text_override")
    if text is None:
        return None
    text = str(text).strip()
    return text or None


def _segment_to_line_payload(segment: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "order": int(segment.get("number", 1)) - 1,
        "kind": LineGeometryKind(segment.get("kind", "polygon")),
        "points": segment["points"],
        "source": LineSource(segment.get("source", "manual")),
        "source_metadata": segment.get("source_metadata"),
        "kraken_ceiling": segment.get("kraken_ceiling"),
        "approved_text": _approved_text(segment),
    }
    if segment.get("baseline") is not None:
        payload["baseline"] = segment["baseline"]
    if segment.get("mask") is not None:
        payload["mask"] = segment["mask"]
    return payload


def _segment_to_history_line(segment: dict[str, Any], line_id: UUID) -> dict[str, Any]:
    return {
        "id": str(line_id),
        "block_id": None,
        "order": int(segment.get("number", 1)) - 1,
        "kind": segment.get("kind", "polygon"),
        "points": segment["points"],
        "source": segment.get("source", "manual"),
        "source_metadata": segment.get("source_metadata"),
        "kraken_ceiling": segment.get("kraken_ceiling"),
        "approved_text": _approved_text(segment),
    }


def _legacy_snapshot_state(annotation: dict[str, Any]) -> dict[str, Any]:
    segments = annotation.get("segments") or []
    lines = [
        _segment_to_history_line(segment, uuid.uuid4())
        for segment in sorted(segments, key=lambda item: int(item.get("number", 1)))
    ]
    return {"lines": lines}


def _parse_legacy_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _discover_page_stems(data_root: Path) -> list[str]:
    pages_dir = data_root / "manuscripts" / "pages"
    annotations_dir = data_root / "annotations" / "pages"
    if not pages_dir.is_dir():
        raise FileNotFoundError(f"Missing page images directory: {pages_dir}")
    if not annotations_dir.is_dir():
        raise FileNotFoundError(f"Missing annotations directory: {annotations_dir}")

    stems: set[str] = set()
    for path in pages_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            stems.add(path.stem)
    for path in annotations_dir.glob("*.json"):
        stems.add(path.stem)
    return sorted(stems)


def _list_legacy_history_files(history_dir: Path, page_stem: str) -> list[Path]:
    page_history_dir = history_dir / page_stem
    if not page_history_dir.is_dir():
        return []
    return sorted(
        (path for path in page_history_dir.glob("*.json") if path.name != "_state.json"),
        key=lambda path: path.stat().st_mtime,
    )


def _image_dimensions(data: bytes) -> tuple[int | None, int | None]:
    try:
        with Image.open(BytesIO(data)) as image:
            return image.size
    except OSError:
        return None, None


async def _ensure_dev_user(session: AsyncSession) -> User:
    auth = AuthService()
    user, _token = await auth.register_if_absent(
        session,
        email=DEV_EMAIL,
        username=DEV_USERNAME,
        password=DEV_PASSWORD,
    )
    if user is None:
        result = await session.execute(select(User).where(User.email == DEV_EMAIL))
        user = result.scalar_one()
    return user


async def _ensure_project(session: AsyncSession, user: User) -> Project:
    result = await session.execute(select(Project).where(Project.slug == PROJECT_SLUG))
    project = result.scalar_one_or_none()
    if project is not None:
        return project
    return await ProjectService().create_project(
        session,
        user,
        name=PROJECT_NAME,
        slug=PROJECT_SLUG,
        guidelines=PROJECT_GUIDELINES,
    )


async def _find_document_by_name(
    session: AsyncSession, project_id: UUID, name: str
) -> Document | None:
    result = await session.execute(
        select(Document).where(Document.project_id == project_id, Document.name == name)
    )
    return result.scalar_one_or_none()


async def _part_has_history(session: AsyncSession, part_id: UUID) -> bool:
    result = await session.execute(
        select(func.count())
        .select_from(AnnotationHistorySnapshot)
        .where(AnnotationHistorySnapshot.part_id == part_id)
    )
    return int(result.scalar_one()) > 0


async def _import_part_history(
    session: AsyncSession,
    *,
    part_id: UUID,
    history_dir: Path,
    page_stem: str,
) -> int:
    """Convert legacy on-disk history snapshots into platform AnnotationHistorySnapshot rows."""
    snapshot_files = _list_legacy_history_files(history_dir, page_stem)
    if not snapshot_files:
        return 0

    imported = 0
    for path in snapshot_files:
        payload = _load_json(path)
        annotation = payload.get("annotation")
        if not isinstance(annotation, dict):
            continue

        state = _legacy_snapshot_state(annotation)
        line_count = len(state["lines"])
        paired_line_count = sum(
            1
            for line in state["lines"]
            if isinstance(line.get("approved_text"), str) and line["approved_text"].strip()
        )
        snapshot = AnnotationHistorySnapshot(
            part_id=part_id,
            state=state,
            line_count=line_count,
            paired_line_count=paired_line_count,
        )
        created_at = _parse_legacy_timestamp(payload.get("timestamp"))
        if created_at is not None:
            snapshot.created_at = created_at
        session.add(snapshot)
        imported += 1

    if imported:
        await session.commit()
    return imported


async def _pair_indexed_segments(
    document_service: DocumentService,
    session: AsyncSession,
    user: User,
    project: Project,
    document: Document,
    part: DocumentPart,
    segments: list[dict[str, Any]],
) -> None:
    indexed = [segment for segment in segments if segment.get("paired_text_line_index") is not None]
    if not indexed:
        return

    lines = await document_service.list_part_lines(
        session, user, project.id, document.id, part.id
    )
    lines_by_order = {line.order: line for line in lines}
    for segment in indexed:
        order = int(segment.get("number", 1)) - 1
        line = lines_by_order.get(order)
        if line is None:
            continue
        await document_service.pair_page_text_line(
            session,
            user,
            project.id,
            document.id,
            part.id,
            line_id=line.id,
            text_line_order=int(segment["paired_text_line_index"]),
        )


async def _import_page_part(
    *,
    session: AsyncSession,
    user: User,
    project: Project,
    document: Document,
    page_stem: str,
    data_root: Path,
    document_service: DocumentService,
    import_history: bool,
    stats: ImportStats,
) -> DocumentPart | None:
    pages_dir = data_root / "manuscripts" / "pages"
    annotations_dir = data_root / "annotations" / "pages"
    transcriptions_dir = data_root / "transcriptions" / "pages"
    history_dir = data_root / "annotations" / "history"

    image_path = _resolve_page_image(pages_dir, page_stem)
    if image_path is None:
        stats.warnings.append(f"{page_stem}: no image in {pages_dir}")
        log.warning("  ! skipping %s: no image file", page_stem)
        return None

    annotation = _load_annotation(annotations_dir, page_stem)
    if annotation is None:
        stats.warnings.append(f"{page_stem}: no annotation JSON")
        log.warning("  ! skipping %s: no annotation JSON", page_stem)
        return None

    image_bytes = image_path.read_bytes()
    part = await document_service.upload_part(
        session,
        user,
        project.id,
        document.id,
        data=image_bytes,
        filename=image_path.name,
    )
    width, height = _image_dimensions(image_bytes)
    if width is not None and height is not None:
        part.width = width
        part.height = height
        await session.commit()
        await session.refresh(part)

    segments = annotation.get("segments") or []
    if segments:
        await document_service.replace_part_lines(
            session,
            user,
            project.id,
            document.id,
            part.id,
            lines=[_segment_to_line_payload(segment) for segment in segments],
        )
        stats.segments += len(segments)

    page_transcription = _load_page_transcription(transcriptions_dir, page_stem)
    if page_transcription is not None:
        await document_service.import_page_transcription(
            session,
            user,
            project.id,
            document.id,
            part.id,
            text=page_transcription,
        )
        await _pair_indexed_segments(
            document_service, session, user, project, document, part, segments
        )

    history_count = 0
    if import_history:
        history_count = await _import_part_history(
            session, part_id=part.id, history_dir=history_dir, page_stem=page_stem
        )
        stats.history_snapshots += history_count

    paired = sum(
        1
        for segment in segments
        if segment.get("text_override") or segment.get("paired_text_line_index") is not None
    )
    log.info(
        "  + part %s: %d segments%s%s",
        page_stem,
        len(segments),
        f", {paired} paired" if paired else "",
        f", {history_count} history snapshot(s)" if history_count else "",
    )
    stats.parts += 1
    return part


async def _backfill_history(
    session: AsyncSession,
    *,
    document: Document,
    page_stems: list[str],
    data_root: Path,
    import_history: bool,
    stats: ImportStats,
) -> None:
    if not import_history:
        return

    history_dir = data_root / "annotations" / "history"
    parts_by_stem = {
        stem.lower(): part
        for part in document.parts
        if (stem := _stem_from_image_key(part.image_key)) is not None
    }

    for page_stem in page_stems:
        part = parts_by_stem.get(page_stem.lower())
        if part is None:
            continue
        if await _part_has_history(session, part.id):
            continue
        history_count = await _import_part_history(
            session, part_id=part.id, history_dir=history_dir, page_stem=page_stem
        )
        if history_count:
            stats.history_snapshots += history_count
            log.info("  + backfilled %d history snapshot(s) for %s", history_count, page_stem)


async def _import_document(
    *,
    session: AsyncSession,
    user: User,
    project: Project,
    doc_name: str,
    page_stems: list[str],
    data_root: Path,
    document_service: DocumentService,
    media: MediaStore,
    force: bool,
    import_history: bool,
) -> ImportStats:
    stats = ImportStats()
    existing = await _find_document_by_name(session, project.id, doc_name)

    if existing is not None and not force:
        stats.documents_skipped = 1
        await session.refresh(existing, attribute_names=["parts"])
        await _backfill_history(
            session,
            document=existing,
            page_stems=page_stems,
            data_root=data_root,
            import_history=import_history,
            stats=stats,
        )
        return stats

    if existing is not None and force:
        for part in existing.parts:
            media.delete(part.image_key)
        await session.delete(existing)
        await session.commit()

    document = await document_service.create_document(
        session, user, project.id, name=doc_name
    )
    stats.documents_created = 1

    for page_stem in page_stems:
        await _import_page_part(
            session=session,
            user=user,
            project=project,
            document=document,
            page_stem=page_stem,
            data_root=data_root,
            document_service=document_service,
            import_history=import_history,
            stats=stats,
        )

    return stats


async def run_seed(*, force: bool, import_history: bool) -> ImportStats:
    if not ANNOTATED_DATA_ROOT.is_dir():
        raise SystemExit(f"Annotated data root not found: {ANNOTATED_DATA_ROOT}")

    page_stems = _discover_page_stems(ANNOTATED_DATA_ROOT)
    if not page_stems:
        raise SystemExit(f"No pages found under {ANNOTATED_DATA_ROOT}")

    document_groups = _group_pages_by_document(page_stems)
    document_service = DocumentService()
    media = MediaStore()
    totals = ImportStats()

    log.info("Data root: %s", ANNOTATED_DATA_ROOT)
    log.info("Pages: %d across %d document(s)", len(page_stems), len(document_groups))
    if import_history:
        log.info("History import: enabled (annotations/history/)")
    else:
        log.info("History import: skipped (--skip-history)")

    async with AsyncSessionLocal() as session:
        user = await _ensure_dev_user(session)
        project = await _ensure_project(session, user)
        log.info("Dev user: %s (id=%s)", DEV_EMAIL, user.id)
        log.info("Project: %s (id=%s)", project.slug, project.id)

        for doc_key, stems in sorted(document_groups.items(), key=lambda item: item[0]):
            doc_name = _document_display_name(doc_key, stems)
            log.info("")
            log.info("Document: %s", doc_name)
            doc_stats = await _import_document(
                session=session,
                user=user,
                project=project,
                doc_name=doc_name,
                page_stems=stems,
                data_root=ANNOTATED_DATA_ROOT,
                document_service=document_service,
                media=media,
                force=force,
                import_history=import_history,
            )
            if doc_stats.documents_skipped:
                log.info("  = already exists (use --force to re-import)")
            totals.merge(doc_stats)

    return totals


def _print_summary(stats: ImportStats) -> None:
    log.info("")
    log.info(
        "Done: %d document(s) imported, %d skipped, %d part(s), %d segment(s), %d history snapshot(s).",
        stats.documents_created,
        stats.documents_skipped,
        stats.parts,
        stats.segments,
        stats.history_snapshots,
    )
    if stats.warnings:
        log.info("Warnings (%d):", len(stats.warnings))
        for warning in stats.warnings:
            log.info("  - %s", warning)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-import documents even when they already exist in the project.",
    )
    parser.add_argument(
        "--skip-history",
        action="store_true",
        help="Do not import legacy annotation history snapshots.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show SQLAlchemy query logging.",
    )
    args = parser.parse_args()

    _configure_logging(verbose=args.verbose)
    stats = asyncio.run(run_seed(force=args.force, import_history=not args.skip_history))
    _print_summary(stats)


if __name__ == "__main__":
    main()
