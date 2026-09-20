"""Compare a local model run with the same run through the platform.

Findings reproduced by the local run (read from the backend, not assumed):

(a) Image bytes: the worker never receives pixels in its claim. It fetches
the single stored page object behind a signed link to ``part.image_key``
(``nomikos/backend/jobs/application/inference_dispatcher.py:180-194``
``page_image_key_for_job``; the link is signed in
``nomikos/backend/jobs/application/job_claim_service.py:186``
``claim_one_page``). An authenticated client fetches exactly those bytes
with ``GET /media/parts/{part_id}`` and no width query
(``nomikos/backend/document/api/media.py:27-46`` ``get_part_image``):
``part_image_response`` serves the stored original when the width is None
and a derived thumbnail rendering otherwise
(``nomikos/backend/document/api/media_responses.py:80-100``,
``nomikos/backend/document/application/part_service.py:448-450``
``_read_part_bytes``). This script always downloads with no width.

(b) Params: an explicit ``model_id`` contributes its catalog
``default_params``; request params are merged over them for segment
(``nomikos/backend/document/application/document_job_enqueue.py:144-175``
``enqueue_segment_part``) while transcribe carries only
``default_params`` plus ``line_ids``
(``nomikos/backend/document/application/document_job_enqueue.py:90-120``
``enqueue_transcribe_part``; request overrides come from a part, document
or project binding in
``nomikos/backend/ml/application/model_service.py:200``
``resolve_for_part``). The dispatcher passes the stored
``job.payload["ml_params"]`` to the worker verbatim for segment
(``nomikos/backend/jobs/application/inference_dispatcher.py:128``
``_build_segment_request``) and as the base params under the line list for
transcribe
(``nomikos/backend/jobs/application/inference_dispatcher.py:162``
``_build_transcribe_request``). This script reuses the finished job
payload ``ml_params`` as the local params, so the merge is reproduced
rather than reimplemented.

(c) Transcribe geometry: the dispatcher sends one entry per line with only
``line_id``, ``line_index`` and ``points``
(``nomikos/backend/jobs/application/inference_dispatcher.py:163-170``),
read in ``(Line.order, Line.created_at)`` order
(``nomikos/backend/document/application/transcribe_merge_service.py:28-35``
``load_lines``). No mask, no baseline, no rounding or int conversion: the
Postgres ``JSONB`` floats
(``nomikos/backend/document/infrastructure/orm_models.py:179``) travel
verbatim into ``TranscribeLineRegion.points``
(``nomikos_inference/contracts/transcribe.py:42-44``). This script builds
the local line list from the same ``GET .../parts/{part_id}/lines``
endpoint, sorted the same way, filtered to the job ``line_ids``.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
import os
import sys
import time
from typing import Any
from urllib import parse as _urlparse
from urllib import request as _urlrequest

try:
    import httpx as _httpx
except ImportError:
    _httpx = None

SEGMENT_REPLACE_FLAG = "i_know_segment_replaces_the_lines"
TERMINAL_STATUSES = ("done", "failed", "cancelled")
_POLY_TOLERANCE_PX = 1.0
_CONFIDENCE_EPS = 1e-4

_SECRET_ENV_NAMES = (
    "NOMIKOS_TOKEN",
    "NOMIKOS_EMAIL",
    "NOMIKOS_PASSWORD",
    "LOCUST_EMAIL",
    "LOCUST_PASSWORD",
)


def _read_env_file(path: str) -> dict[str, str]:
    values: dict[str, str] = {}
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key.startswith("export "):
                key = key[len("export ") :].strip()
            if key:
                values[key] = value
    return values


def _credentials(args: argparse.Namespace) -> tuple[str | None, str | None, str | None]:
    file_values = _read_env_file(args.env_file) if args.env_file else {}
    token = os.environ.get("NOMIKOS_TOKEN", file_values.get("NOMIKOS_TOKEN", ""))
    email = os.environ.get(
        "NOMIKOS_EMAIL",
        file_values.get(
            "NOMIKOS_EMAIL", file_values.get("LOCUST_EMAIL", os.environ.get("LOCUST_EMAIL", ""))
        ),
    )
    password = os.environ.get(
        "NOMIKOS_PASSWORD",
        file_values.get(
            "NOMIKOS_PASSWORD",
            file_values.get("LOCUST_PASSWORD", os.environ.get("LOCUST_PASSWORD", "")),
        ),
    )
    return (token or None, email or None, password or None)


def _scrub(value: Any, secrets: list[str]) -> Any:
    if isinstance(value, dict):
        return {key: _scrub(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [_scrub(item, secrets) for item in value]
    if isinstance(value, str):
        cleaned = value
        for secret in secrets:
            if secret:
                cleaned = cleaned.replace(secret, "[REDACTED]")
        return cleaned
    return value


def parse_artifact_ref(artifact_ref: str) -> tuple[str, str]:
    parsed = _urlparse.urlparse(artifact_ref)
    if parsed.scheme != "registry" or not parsed.netloc:
        raise ValueError(f"unsupported artifact_ref: {artifact_ref!r}")
    query = _urlparse.parse_qs(parsed.query)
    tag = (query.get("tag", ["stable"])[0] or "stable").strip() or "stable"
    return parsed.netloc, tag


def resolve_model(catalog: list[dict[str, Any]], spec: str) -> dict[str, Any]:
    wanted = spec.strip()
    for entry in catalog:
        if str(entry.get("id", "")) == wanted:
            return entry
    for entry in catalog:
        ref = str(entry.get("artifact_ref", ""))
        try:
            registry_id, _ = parse_artifact_ref(ref)
        except ValueError:
            continue
        if registry_id == wanted:
            return entry
    lowered = wanted.lower()
    for entry in catalog:
        if str(entry.get("name", "")).lower() == lowered:
            return entry
    raise ValueError(f"model {spec!r} not found in catalog of {len(catalog)} entries")


def _as_floats(points: Any) -> list[list[float]]:
    out: list[list[float]] = []
    if not isinstance(points, list):
        return out
    for point in points:
        if isinstance(point, (list, tuple)) and len(point) == 2:
            try:
                out.append([float(point[0]), float(point[1])])
            except (TypeError, ValueError):
                continue
    return out


def _geom_points(geom: Any) -> list[list[float]]:
    if geom is None:
        return []
    if isinstance(geom, dict):
        for key in ("points", "poly", "polygon"):
            if key in geom:
                return _as_floats(geom[key])
        return []
    return _as_floats(geom)


def max_coord_diff(first: list[list[float]], second: list[list[float]]) -> float | None:
    if len(first) != len(second):
        return None
    if not first:
        return 0.0
    diff = 0.0
    for (x1, y1), (x2, y2) in zip(first, second, strict=True):
        diff = max(diff, abs(x1 - x2), abs(y1 - y2))
    return diff


def _polygon_area(points: list[list[float]]) -> float:
    total = 0.0
    count = len(points)
    for index in range(count):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % count]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def _clip_half_plane(
    polygon: list[list[float]], edge: int, bound: float, keep_below: bool
) -> list[list[float]]:
    if not polygon:
        return []
    axis = edge % 2
    out: list[list[float]] = []
    count = len(polygon)
    for index in range(count):
        current = polygon[index]
        previous = polygon[index - 1]
        c_in = current[axis] <= bound if keep_below else current[axis] >= bound
        p_in = previous[axis] <= bound if keep_below else previous[axis] >= bound
        if c_in:
            if not p_in:
                denom = current[axis] - previous[axis]
                ratio = (bound - previous[axis]) / denom if denom else 0.0
                out.append(
                    [
                        previous[0] + ratio * (current[0] - previous[0]),
                        previous[1] + ratio * (current[1] - previous[1]),
                    ]
                )
            out.append(current)
        elif p_in:
            denom = current[axis] - previous[axis]
            ratio = (bound - previous[axis]) / denom if denom else 0.0
            out.append(
                [
                    previous[0] + ratio * (current[0] - previous[0]),
                    previous[1] + ratio * (current[1] - previous[1]),
                ]
            )
    return out


def polygon_intersection_area(first: list[list[float]], second: list[list[float]]) -> float:
    if len(first) < 3 or len(second) < 3:
        return 0.0
    xs = [point[0] for point in second]
    ys = [point[1] for point in second]
    clipped = [list(point) for point in first]
    for edge, bound, keep_below in (
        (0, min(xs), False),
        (0, max(xs), True),
        (1, min(ys), False),
        (1, max(ys), True),
    ):
        clipped = _clip_half_plane(clipped, edge, bound, keep_below)
        if len(clipped) < 3:
            return 0.0
    return _polygon_area(clipped)


def polygon_iou(first: list[list[float]], second: list[list[float]]) -> float:
    area_first = _polygon_area(first)
    area_second = _polygon_area(second)
    if area_first <= 0.0 or area_second <= 0.0:
        return 0.0
    inter = polygon_intersection_area(first, second)
    union = area_first + area_second - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _norm_segment_line(line: dict[str, Any]) -> dict[str, Any]:
    meta = line.get("source_metadata") or {}
    return {
        "order": line.get("order"),
        "kind": line.get("kind"),
        "points": _as_floats(line.get("points")),
        "baseline": _geom_points(line.get("baseline")),
        "mask": _geom_points(line.get("mask")),
        "role": meta.get("role"),
        "suspect": meta.get("suspect"),
        "external_id": line.get("external_id"),
    }


def compare_segment_lines(
    local_lines: list[dict[str, Any]], platform_lines: list[dict[str, Any]]
) -> dict[str, Any]:
    local = [_norm_segment_line(line) for line in local_lines]
    platform = [_norm_segment_line(line) for line in platform_lines]
    local_by_order = sorted(local, key=lambda line: (line["order"] is None, line["order"]))
    platform_by_order = sorted(platform, key=lambda line: (line["order"] is None, line["order"]))
    result: dict[str, Any] = {
        "local_line_count": len(local),
        "platform_line_count": len(platform),
        "verdict": "IDENTICAL",
        "order_pairing": [],
        "iou_pairing": [],
        "only_local": [],
        "only_platform": [],
        "order_differences": [],
        "line_differences": [],
    }
    if len(local) != len(platform):
        result["verdict"] = "MISMATCH"
    pairs = list(zip(local_by_order, platform_by_order, strict=False))
    max_diff = 0.0
    for position, (left, right) in enumerate(pairs):
        entry: dict[str, Any] = {
            "position": position,
            "local_order": left["order"],
            "platform_order": right["order"],
            "points_diff": max_coord_diff(left["points"], right["points"]),
            "baseline_diff": max_coord_diff(left["baseline"], right["baseline"]),
            "mask_diff": max_coord_diff(left["mask"], right["mask"]),
            "kind_equal": left["kind"] == right["kind"],
            "role_equal": left["role"] == right["role"],
            "suspect_equal": left["suspect"] == right["suspect"],
        }
        result["order_pairing"].append(entry)
        problems: list[str] = []
        if left["order"] != right["order"]:
            problems.append("order")
            result["order_differences"].append(entry)
        for key in ("points_diff", "baseline_diff", "mask_diff"):
            value = entry[key]
            if value is None:
                problems.append(key)
            else:
                max_diff = max(max_diff, value)
                if value > _POLY_TOLERANCE_PX:
                    problems.append(key)
        if not entry["kind_equal"]:
            problems.append("kind")
        if not entry["role_equal"]:
            problems.append("role")
        if not entry["suspect_equal"]:
            problems.append("suspect")
        if problems:
            result["line_differences"].append({"position": position, "fields": problems})
    matched_platform: set[int] = set()
    for local_index, left in enumerate(local):
        best_index = -1
        best_iou = 0.0
        for platform_index, right in enumerate(platform):
            if platform_index in matched_platform:
                continue
            score = polygon_iou(left["points"], right["points"])
            if score > best_iou:
                best_iou = score
                best_index = platform_index
        if best_index >= 0 and best_iou > 0.0:
            matched_platform.add(best_index)
            result["iou_pairing"].append(
                {"local": local_index, "platform": best_index, "iou": best_iou}
            )
            partner = platform[best_index]
            if (
                best_iou >= 0.5
                and left["order"] is not None
                and partner["order"] is not None
                and left["order"] != partner["order"]
            ):
                result["order_differences"].append(
                    {
                        "local": local_index,
                        "platform": best_index,
                        "local_order": left["order"],
                        "platform_order": partner["order"],
                        "iou": best_iou,
                    }
                )
        else:
            result["only_local"].append(local_index)
    result["only_platform"] = [
        index for index in range(len(platform)) if index not in matched_platform
    ]
    if result["only_local"] or result["only_platform"] or result["line_differences"]:
        result["verdict"] = "MISMATCH"
    elif max_diff > 0.0:
        result["verdict"] = "NUMERIC"
    result["max_coordinate_diff"] = max_diff
    return result


def _levenshtein(first: str, second: str) -> int:
    if len(first) < len(second):
        first, second = second, first
    previous = list(range(len(second) + 1))
    for index, char_first in enumerate(first, start=1):
        current = [index]
        for other, char_second in enumerate(second, start=1):
            cost = 0 if char_first == char_second else 1
            current.append(
                min(previous[other] + 1, current[other - 1] + 1, previous[other - 1] + cost)
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    if not reference and not hypothesis:
        return 0.0
    if not reference:
        return 1.0
    return _levenshtein(reference, hypothesis) / len(reference)


def _norm_transcribe_line(line: dict[str, Any]) -> dict[str, Any]:
    output = line.get("output") or {}
    return {
        "line_id": line.get("line_id"),
        "line_index": line.get("line_index"),
        "text": output.get("text", ""),
        "confidence": output.get("confidence"),
        "error": line.get("error"),
    }


def compare_transcribe_lines(
    local_lines: list[dict[str, Any]], platform_lines: list[dict[str, Any]]
) -> dict[str, Any]:
    local = [_norm_transcribe_line(line) for line in local_lines]
    platform = [_norm_transcribe_line(line) for line in platform_lines]
    by_id_local = {line["line_id"]: line for line in local if line["line_id"] is not None}
    by_id_platform = {line["line_id"]: line for line in platform if line["line_id"] is not None}
    result: dict[str, Any] = {
        "local_line_count": len(local),
        "platform_line_count": len(platform),
        "verdict": "IDENTICAL",
        "line_differences": [],
        "max_confidence_diff": 0.0,
        "max_cer": 0.0,
        "only_local": [],
        "only_platform": [],
    }
    if len(local) != len(platform):
        result["verdict"] = "MISMATCH"
    if by_id_local and len(by_id_local) == len(local) and len(by_id_platform) == len(platform):
        pairs = [
            (line, by_id_platform.get(line["line_id"]))
            for line in sorted(local, key=lambda item: item["line_index"] or 0)
        ]
    else:
        ordered_local = sorted(local, key=lambda item: item["line_index"] or 0)
        ordered_platform = sorted(platform, key=lambda item: item["line_index"] or 0)
        pairs = list(zip(ordered_local, ordered_platform, strict=False))
    for left, right in pairs:
        if right is None:
            result["only_local"].append(left["line_id"])
            result["verdict"] = "MISMATCH"
            continue
        left_text = left["text"] or ""
        right_text = right["text"] or ""
        cer = character_error_rate(right_text, left_text)
        result["max_cer"] = max(result["max_cer"], cer)
        conf_diff: float | None = None
        if left["confidence"] is not None and right["confidence"] is not None:
            conf_diff = abs(float(left["confidence"]) - float(right["confidence"]))
            result["max_confidence_diff"] = max(result["max_confidence_diff"], conf_diff)
        entry: dict[str, Any] = {
            "line_id": left["line_id"],
            "line_index": left["line_index"],
            "local_text": left_text,
            "platform_text": right_text,
            "text_equal": left_text == right_text,
            "cer": cer,
            "confidence_diff": conf_diff,
            "local_error": left["error"],
            "platform_error": right["error"],
        }
        if (
            not entry["text_equal"]
            or (conf_diff is not None and conf_diff > _CONFIDENCE_EPS)
            or (left["error"] or None) != (right["error"] or None)
        ):
            result["line_differences"].append(entry)
    local_ids = {line["line_id"] for line in local}
    platform_ids = {line["line_id"] for line in platform}
    result["only_platform"] = sorted(
        str(item) for item in (platform_ids - local_ids) if item is not None
    )
    if result["only_local"] or result["only_platform"]:
        result["verdict"] = "MISMATCH"
    elif result["line_differences"]:
        texts_equal = all(item["text_equal"] for item in result["line_differences"])
        if texts_equal:
            result["verdict"] = "CONFIDENCE_ONLY"
        else:
            result["verdict"] = "MISMATCH"
    return result


class ApiError(Exception):
    pass


class ApiClient:
    """Small API client with an injectable transport for offline tests.

    The transport is a callable ``(method, url, headers, body)`` returning
    ``(status_code, response_body_bytes)``. When None, a default transport
    backed by httpx (or urllib when httpx is unavailable) is used.
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        transport: Any = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self._transport = transport or self._default_transport

    def _default_transport(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None
    ) -> tuple[int, bytes]:
        if _httpx is not None:
            with _httpx.Client(timeout=60.0) as client:
                response = client.request(method, url, headers=headers, content=body)
                return response.status_code, response.content
        data = body if method in ("POST", "PUT", "PATCH") else None
        request = _urlrequest.Request(  # noqa: S310
            url, data=data, method=method, headers=headers or {}
        )
        try:
            with _urlrequest.urlopen(request, timeout=60) as response:  # noqa: S310
                return response.status, response.read()
        except Exception as error:
            message = str(error)
            if hasattr(error, "read"):
                try:
                    payload = error.read()
                    detail = json.loads(payload.decode("utf-8"))
                    raise ApiError(f"{method} {url} failed: {detail}") from error
                except (ValueError, UnicodeDecodeError):
                    pass
            raise ApiError(f"{method} {url} failed: {message}") from error

    def _headers(self, as_json: bool) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self.token}"}
        if as_json:
            headers["Content-Type"] = "application/json"
        return headers

    def request_json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        url = self.base_url + path
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        status, raw = self._transport(method, url, self._headers(payload is not None), body)
        if status >= 400:
            raise ApiError(f"{method} {path} returned {status}: {raw[:500]!r}")
        if not raw:
            return None
        return json.loads(raw.decode("utf-8"))

    def request_bytes(self, method: str, path: str) -> bytes:
        url = self.base_url + path
        status, raw = self._transport(method, url, self._headers(False), None)
        if status >= 400:
            raise ApiError(f"{method} {path} returned {status}")
        return raw

    def login(self, email: str, password: str) -> str:
        url = self.base_url + "/auth/login"
        body = json.dumps({"email": email, "password": password}).encode("utf-8")
        status, raw = self._transport("POST", url, {"Content-Type": "application/json"}, body)
        if status >= 400:
            raise ApiError(f"POST /auth/login returned {status}")
        data = json.loads(raw.decode("utf-8"))
        return str(data["access_token"])

    def list_models(self) -> list[dict[str, Any]]:
        data = self.request_json("GET", "/inference/models")
        return list(data) if isinstance(data, list) else []

    def get_job(self, job_id: str) -> dict[str, Any]:
        data = self.request_json("GET", f"/jobs/{job_id}")
        if not isinstance(data, dict):
            raise ApiError(f"GET /jobs/{job_id} returned a non-object")
        return data

    def list_part_lines(
        self, project_id: str, document_id: str, part_id: str
    ) -> list[dict[str, Any]]:
        data = self.request_json(
            "GET", f"/projects/{project_id}/documents/{document_id}/parts/{part_id}/lines"
        )
        return list(data) if isinstance(data, list) else []

    def page_image_bytes(self, part_id: str) -> bytes:
        return self.request_bytes("GET", f"/media/parts/{part_id}")

    def enqueue_segment(
        self, project_id: str, document_id: str, part_id: str, model_id: str
    ) -> dict[str, Any]:
        data = self.request_json(
            "POST",
            f"/projects/{project_id}/documents/{document_id}/parts/{part_id}/segment",
            {"model_id": model_id},
        )
        if not isinstance(data, dict):
            raise ApiError("segment enqueue returned a non-object")
        return data

    def enqueue_transcribe(
        self,
        project_id: str,
        document_id: str,
        part_id: str,
        model_id: str,
        line_ids: list[str] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model_id": model_id}
        if line_ids is not None:
            payload["line_ids"] = line_ids
        data = self.request_json(
            "POST",
            f"/projects/{project_id}/documents/{document_id}/parts/{part_id}/transcribe",
            payload,
        )
        if not isinstance(data, dict):
            raise ApiError("transcribe enqueue returned a non-object")
        return data


def local_package_versions() -> dict[str, str | None]:
    nomikos_version: str | None = None
    try:
        from importlib import metadata as _metadata

        nomikos_version = _metadata.version("nomikos-inference")
    except Exception:
        try:
            import nomikos_inference as _package

            nomikos_version = getattr(_package, "__version__", None)
        except Exception:
            nomikos_version = None
    onnx_version: str | None = None
    try:
        import onnxruntime as _ort

        onnx_version = getattr(_ort, "__version__", None)
    except Exception:
        onnx_version = None
    return {"nomikos_inference": nomikos_version, "onnxruntime": onnx_version}


def _model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return [_model_dump(item) for item in value]
    return value


def platform_result_lines(task: str, result: Any) -> list[dict[str, Any]]:
    if not isinstance(result, dict):
        return []
    if task == "segment":
        lines = result.get("lines")
        return list(lines) if isinstance(lines, list) else []
    lines = result.get("lines")
    if isinstance(lines, list):
        return list(lines)
    return []


def worker_version_from_job(job: dict[str, Any]) -> str | None:
    for container in (job.get("result"), job.get("payload"), job.get("metadata")):
        if isinstance(container, dict):
            for key in (
                "worker_version",
                "nomikos_inference_version",
                "inference_version",
                "artifact_sha256",
            ):
                value = container.get(key)
                if isinstance(value, str) and value:
                    return value
    return None


def build_transcribe_params(
    part_lines: list[dict[str, Any]],
    job_line_ids: list[str] | None,
    base_params: dict[str, Any],
    line_limit: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ordered = sorted(
        part_lines,
        key=lambda line: (
            line.get("order") if isinstance(line.get("order"), int) else 0,
            str(line.get("created_at") or ""),
        ),
    )
    if job_line_ids:
        wanted = set(job_line_ids)
        ordered = [line for line in ordered if str(line.get("id")) in wanted]
    if line_limit is not None:
        ordered = ordered[:line_limit]
    regions = []
    for index, line in enumerate(ordered):
        regions.append(
            {
                "line_id": str(line.get("id")),
                "line_index": index,
                "points": line.get("points"),
            }
        )
    params = dict(base_params)
    params.pop("lines", None)
    params["lines"] = regions
    return params, ordered


def render_report_markdown(report: dict[str, Any]) -> str:
    header = report["header"]
    comparison = report["comparison"]
    lines = [
        "# Parity report: local vs platform",
        "",
        f"Task: {header['task']}",
        f"Model: {header['model_name']} (registry {header['registry_model_id']}, "
        f"tag {header['registry_tag']})",
        f"Job: {header['job_id']}",
        f"Verdict: {comparison['verdict']}",
        "",
        "## Header",
        "",
        f"API host: {header['api_host']}",
        f"Local nomikos_inference: {header['local_nomikos_inference_version']}",
        f"Local onnxruntime: {header['local_onnxruntime_version']}",
        f"Worker version: {header['worker_version']}",
        f"Image sha256: {header['image_sha256']} ({header['image_bytes']} bytes)",
        f"Params: {json.dumps(header['params'], sort_keys=True)}",
        "",
        "## Comparison",
        "",
        f"Local lines: {comparison.get('local_line_count')}",
        f"Platform lines: {comparison.get('platform_line_count')}",
    ]
    if header["task"] == "segment":
        lines.append(f"Max coordinate diff (px): {comparison.get('max_coordinate_diff')}")
        if comparison.get("order_differences"):
            lines.append(f"Order differences: {len(comparison['order_differences'])}")
        for item in comparison.get("line_differences", [])[:50]:
            lines.append(f"Line {item['position']}: {', '.join(item['fields'])}")
        if comparison.get("only_local"):
            lines.append(f"Only local: {comparison['only_local']}")
        if comparison.get("only_platform"):
            lines.append(f"Only platform: {comparison['only_platform']}")
    else:
        lines.append(f"Max CER: {comparison.get('max_cer')}")
        lines.append(f"Max confidence diff: {comparison.get('max_confidence_diff')}")
        for item in comparison.get("line_differences", [])[:50]:
            lines.append(
                f"Line {item.get('line_id')} (index {item.get('line_index')}): "
                f"local {item.get('local_text')!r} platform {item.get('platform_text')!r} "
                f"CER {item.get('cer')}"
            )
        if comparison.get("only_local"):
            lines.append(f"Only local: {comparison['only_local']}")
        if comparison.get("only_platform"):
            lines.append(f"Only platform: {comparison['only_platform']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare a local nomikos_inference run with the same platform job."
    )
    parser.add_argument("--api", default="https://api.nomikos.app")
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--project-id", default=None)
    parser.add_argument("--document-id", default=None)
    parser.add_argument("--part-id", default=None)
    parser.add_argument("--task", choices=("segment", "transcribe"), default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--enqueue", action="store_true")
    parser.add_argument("--i-know-segment-replaces-the-lines", action="store_true")
    parser.add_argument("--line-limit", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser


def poll_job(client: ApiClient, job_id: str, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while True:
        job = client.get_job(job_id)
        if str(job.get("status")) in TERMINAL_STATUSES:
            return job
        if time.monotonic() >= deadline:
            raise ApiError(f"timed out waiting for job {job_id}")
        time.sleep(5.0)


def _default_out_dir() -> str:
    stamp = _datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join("/tmp", "nmk-parity", stamp)  # noqa: S108


def collect_secret_values(args: argparse.Namespace) -> list[str]:
    secrets: list[str] = []
    for name in _SECRET_ENV_NAMES:
        value = os.environ.get(name)
        if value:
            secrets.append(value)
    if args.env_file:
        try:
            file_values = _read_env_file(args.env_file)
        except OSError:
            file_values = {}
        for name in _SECRET_ENV_NAMES:
            value = file_values.get(name)
            if value:
                secrets.append(value)
    return [secret for secret in secrets if secret]


def run_comparison(
    args: argparse.Namespace,
    client: ApiClient,
    out_dir: str,
) -> int:
    if args.task == "segment" and args.enqueue and not args.i_know_segment_replaces_the_lines:
        print(
            "Refusing: a segment job replaces the part lines. "
            "Re-run with --i-know-segment-replaces-the-lines to proceed.",
            file=sys.stderr,
        )
        return 2
    catalog = client.list_models()
    if args.model is None:
        raise ApiError("missing required --model")
    model_entry = resolve_model(catalog, args.model)
    registry_model_id, registry_tag = parse_artifact_ref(str(model_entry["artifact_ref"]))
    model_uuid = str(model_entry.get("id"))

    job_id = args.job_id
    if args.enqueue:
        if not (args.project_id and args.document_id and args.part_id):
            raise ApiError("--enqueue needs --project-id, --document-id and --part-id")
        if args.task == "segment":
            current = client.list_part_lines(args.project_id, args.document_id, args.part_id)
            print(f"Part currently has {len(current)} lines; segment will replace them.")
            created = client.enqueue_segment(
                args.project_id, args.document_id, args.part_id, model_uuid
            )
        else:
            created = client.enqueue_transcribe(
                args.project_id, args.document_id, args.part_id, model_uuid, None
            )
        job_id = str(created.get("job_id") or created.get("id"))
        if not job_id or job_id == "None":
            raise ApiError(f"enqueue response has no job id: {created!r}")
    if not job_id:
        raise ApiError("missing required --job-id (or use --enqueue to create one)")
    job = poll_job(client, job_id, args.timeout)
    if str(job.get("status")) == "failed":
        print(f"Platform job {job_id} failed: {job.get('error')}", file=sys.stderr)
        return 2
    if str(job.get("status")) != "done":
        print(
            f"Platform job {job_id} ended with status {job.get('status')}",
            file=sys.stderr,
        )
        return 2
    part_id = str(job.get("document_part_id") or args.part_id or "")
    if not part_id:
        raise ApiError("job has no document_part_id and --part-id was not given")
    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    base_params = payload.get("ml_params") if isinstance(payload.get("ml_params"), dict) else {}
    params: dict[str, Any] = dict(base_params)
    project_id = str(job.get("project_id") or args.project_id or "")
    document_id = str(job.get("document_id") or args.document_id or "")
    if args.task == "transcribe":
        if not (project_id and document_id):
            raise ApiError("transcribe needs --project-id and --document-id")
        part_lines = client.list_part_lines(project_id, document_id, part_id)
        job_line_ids = payload.get("line_ids")
        params, _ = build_transcribe_params(
            part_lines,
            list(job_line_ids) if isinstance(job_line_ids, list) else None,
            base_params,
            args.line_limit,
        )

    image_bytes = client.page_image_bytes(part_id)
    image_sha256 = hashlib.sha256(image_bytes).hexdigest()

    from nomikos_inference.contracts.common import InferenceTask
    from nomikos_inference.jobs.runner import run_model

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    local_response = run_model(
        task=InferenceTask(args.task),
        registry_model_id=registry_model_id,
        registry_tag=registry_tag,
        image_bytes=image_bytes,
        params=params,
    )
    local_dump = _model_dump(local_response)
    platform_result = job.get("result") if isinstance(job.get("result"), dict) else {}
    if args.task == "segment":
        comparison = compare_segment_lines(
            local_dump.get("lines", []), platform_result_lines("segment", platform_result)
        )
    else:
        comparison = compare_transcribe_lines(
            local_dump.get("lines", []), platform_result_lines("transcribe", platform_result)
        )
    versions = local_package_versions()
    secrets = collect_secret_values(args)
    header = {
        "api_host": args.api,
        "task": args.task,
        "model_name": model_entry.get("name"),
        "registry_model_id": registry_model_id,
        "registry_tag": registry_tag,
        "model_id": model_uuid,
        "job_id": job_id,
        "local_nomikos_inference_version": versions["nomikos_inference"],
        "local_onnxruntime_version": versions["onnxruntime"],
        "worker_version": worker_version_from_job(job),
        "params": params,
        "image_sha256": image_sha256,
        "image_bytes": len(image_bytes),
    }
    report: dict[str, Any] = {
        "header": header,
        "comparison": comparison,
        "local_result": local_dump,
        "platform_result": platform_result,
    }
    report = _scrub(report, secrets)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=str)
    markdown = render_report_markdown(report)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as handle:
        handle.write(markdown)
    print(f"Verdict: {comparison['verdict']} (report in {out_dir})")
    return 0 if comparison["verdict"] == "IDENTICAL" else 1


def main(argv: list[str] | None = None, transport: Any = None) -> int:
    args = build_parser().parse_args(argv)
    if args.task is None:
        print("missing required --task (segment|transcribe)", file=sys.stderr)
        return 2
    if args.task == "segment" and args.enqueue and not args.i_know_segment_replaces_the_lines:
        print(
            "Refusing: a segment job replaces the part lines. "
            "Re-run with --i-know-segment-replaces-the-lines to proceed.",
            file=sys.stderr,
        )
        return 2
    token, email, password = _credentials(args)
    if token is None:
        if email is None or password is None:
            print(
                "Missing credentials: set NOMIKOS_TOKEN or "
                "NOMIKOS_EMAIL plus NOMIKOS_PASSWORD (or --env-file).",
                file=sys.stderr,
            )
            return 2
        client = ApiClient(args.api, "", transport=transport)
        token = client.login(email, password)
    client = ApiClient(args.api, token, transport=transport)
    out_dir = args.out or _default_out_dir()
    try:
        return run_comparison(args, client, out_dir)
    except ApiError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
