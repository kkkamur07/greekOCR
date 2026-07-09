"""Pull registry.yaml from the hosted API into a local cache."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import httpx

from inference.registry import load_registry

logger = logging.getLogger(__name__)


def sync_registry_from_url(
    url: str,
    *,
    cached_path: Path,
    etag_path: Path,
    fallback_path: Path,
    timeout_seconds: float = 15.0,
) -> Path:
    """Fetch registry when online; return cached or bundled path on failure."""
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    stored_etag = etag_path.read_text(encoding="utf-8").strip() if etag_path.exists() else None
    headers: dict[str, str] = {}
    if stored_etag:
        quoted_etag = stored_etag if stored_etag.startswith('"') else f'"{stored_etag}"'
        headers["If-None-Match"] = quoted_etag

    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.get(url, headers=headers)
    except httpx.HTTPError as exc:
        logger.warning("Registry sync failed for %s: %s", url, exc)
        return _resolve_registry_path(cached_path, fallback_path)

    if response.status_code == 304:
        if cached_path.exists():
            return cached_path
        logger.warning("Registry sync returned 304 but %s is missing", cached_path)
        return _resolve_registry_path(cached_path, fallback_path)

    if response.status_code != 200:
        logger.warning("Registry sync HTTP %s from %s", response.status_code, url)
        return _resolve_registry_path(cached_path, fallback_path)

    content = response.text
    etag = response.headers.get("etag", "").strip().strip('"') or _registry_etag(content)
    try:
        _validate_and_write(content, cached_path, etag_path, etag)
    except ValueError as exc:
        logger.warning("Registry sync rejected invalid payload: %s", exc)
        return _resolve_registry_path(cached_path, fallback_path)

    logger.info("Registry synced from %s", url)
    return cached_path


def _registry_etag(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _validate_and_write(content: str, cached_path: Path, etag_path: Path, etag: str) -> None:
    tmp_path = cached_path.with_suffix(".yaml.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    try:
        load_registry(tmp_path)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(str(exc)) from exc
    tmp_path.replace(cached_path)
    etag_path.write_text(etag, encoding="utf-8")


def _resolve_registry_path(cached_path: Path, fallback_path: Path) -> Path:
    if cached_path.exists():
        return cached_path
    return fallback_path
