"""Unit tests for the local vs platform parity harness.

No network and no model weights: the API transport is faked and the
runner is stubbed where the full flow is exercised.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "inference"
    / "parity"
    / "compare_local_vs_platform.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("nmk_parity", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nmk = load_module()

FAKE_CATALOG = [
    {
        "id": "11111111-1111-1111-1111-111111111111",
        "name": "ppocr",
        "task": "segment",
        "artifact_ref": "registry://ppocr-segment?tag=stable",
        "default_params": {},
    },
    {
        "id": "22222222-2222-2222-2222-222222222222",
        "name": "syriac",
        "task": "transcribe",
        "artifact_ref": "registry://syriac-ppocr-v1?tag=stable",
        "default_params": {"beam": 3},
    },
]


def seg_line(order, points, baseline=None, kind="polygon"):
    return {
        "external_id": f"line-{order}",
        "order": order,
        "kind": kind,
        "points": points,
        "baseline": {"points": baseline} if baseline is not None else {"points": []},
        "mask": None,
        "source_metadata": {"role": "line", "suspect": False},
    }


BOX_A = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
BOX_B = [[0.0, 20.0], [10.0, 20.0], [10.0, 30.0], [0.0, 30.0]]
BASE_A = [[0.0, 7.0], [10.0, 7.0]]


def shift(points, delta):
    return [[x + delta, y + delta] for x, y in points]


def test_resolve_model_by_display_name_registry_id_and_uuid():
    by_name = nmk.resolve_model(FAKE_CATALOG, "ppocr")
    assert by_name["artifact_ref"] == "registry://ppocr-segment?tag=stable"
    by_registry = nmk.resolve_model(FAKE_CATALOG, "ppocr-segment")
    assert by_registry["name"] == "ppocr"
    by_uuid = nmk.resolve_model(FAKE_CATALOG, "11111111-1111-1111-1111-111111111111")
    assert by_uuid["name"] == "ppocr"
    registry_id, tag = nmk.parse_artifact_ref(by_name["artifact_ref"])
    assert (registry_id, tag) == ("ppocr-segment", "stable")
    with pytest.raises(ValueError):
        nmk.resolve_model(FAKE_CATALOG, "no-such-model")


def test_segment_identical():
    lines = [seg_line(0, BOX_A, BASE_A), seg_line(1, BOX_B)]
    result = nmk.compare_segment_lines(lines, [dict(line) for line in lines])
    assert result["verdict"] == "IDENTICAL"
    assert result["local_line_count"] == result["platform_line_count"] == 2


def test_segment_numeric_shift():
    local = [seg_line(0, BOX_A, BASE_A)]
    platform = [seg_line(0, shift(BOX_A, 0.5), shift(BASE_A, 0.5))]
    result = nmk.compare_segment_lines(local, platform)
    assert result["verdict"] == "NUMERIC"
    assert result["max_coordinate_diff"] == pytest.approx(0.5)


def test_segment_reordered_is_mismatch_with_order_differences():
    local = [seg_line(0, BOX_A, BASE_A), seg_line(1, BOX_B)]
    platform = [seg_line(0, BOX_B), seg_line(1, BOX_A, BASE_A)]
    result = nmk.compare_segment_lines(local, platform)
    assert result["verdict"] == "MISMATCH"
    assert result["order_differences"]


def test_segment_missing_line():
    local = [seg_line(0, BOX_A, BASE_A), seg_line(1, BOX_B)]
    platform = [seg_line(0, BOX_A, BASE_A)]
    result = nmk.compare_segment_lines(local, platform)
    assert result["verdict"] == "MISMATCH"
    assert result["only_local"] == [1]
    assert result["platform_line_count"] == 1


def tr_line(line_id, index, text, confidence=0.9):
    return {
        "line_id": line_id,
        "line_index": index,
        "output": {"text": text, "confidence": confidence, "character_confidences": []},
    }


def test_transcribe_identical():
    lines = [tr_line("a", 0, "hello"), tr_line("b", 1, "world")]
    result = nmk.compare_transcribe_lines(lines, [dict(line) for line in lines])
    assert result["verdict"] == "IDENTICAL"


def test_transcribe_confidence_only():
    local = [tr_line("a", 0, "hello", 0.9)]
    platform = [tr_line("a", 0, "hello", 0.5)]
    result = nmk.compare_transcribe_lines(local, platform)
    assert result["verdict"] == "CONFIDENCE_ONLY"
    assert result["max_confidence_diff"] == pytest.approx(0.4)


def test_transcribe_text_diff_lists_both_texts():
    local = [tr_line("a", 0, "hello")]
    platform = [tr_line("a", 0, "hallo")]
    result = nmk.compare_transcribe_lines(local, platform)
    assert result["verdict"] == "MISMATCH"
    assert len(result["line_differences"]) == 1
    diff = result["line_differences"][0]
    assert diff["local_text"] == "hello"
    assert diff["platform_text"] == "hallo"
    assert result["max_cer"] > 0.0


class FakeTransport:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def __call__(self, method, url, headers, body):
        self.calls.append((method, url))
        bare = url.split("?", 1)[0]
        for (route_method, route_path), handler in self.routes.items():
            if method == route_method and bare.endswith(route_path):
                return handler(method, url, body)
        raise AssertionError(f"unexpected request {method} {url}")


def json_route(payload, status=200):
    def handler(method, url, body):
        return status, json.dumps(payload).encode("utf-8")

    return handler


def test_segment_enqueue_without_flag_makes_no_network_call():
    transport = FakeTransport({})
    code = nmk.main(
        [
            "--task",
            "segment",
            "--enqueue",
            "--project-id",
            "x",
            "--document-id",
            "y",
            "--part-id",
            "z",
            "--model",
            "ppocr",
            "--api",
            "http://127.0.0.1:9",
        ],
        transport=transport,
    )
    assert code != 0
    assert transport.calls == []


def test_secrets_never_appear_in_report(tmp_path, monkeypatch):
    secret = "super-secret-token-value-xyz"
    monkeypatch.setenv("NOMIKOS_TOKEN", secret)
    job = {
        "id": "job-1",
        "status": "done",
        "error": None,
        "document_part_id": "part-1",
        "project_id": "proj-1",
        "document_id": "doc-1",
        "payload": {"ml_params": {"note": secret}},
        "result": {"blocks": [], "lines": []},
    }
    routes = {
        ("GET", "/inference/models"): json_route(FAKE_CATALOG),
        ("GET", "/jobs/job-1"): json_route(job),
        ("GET", "/media/parts/part-1"): lambda m, u, b: (200, b"fake-bytes"),
    }
    transport = FakeTransport(routes)

    import nomikos_inference.jobs.runner as runner_module

    class FakeResponse:
        def model_dump(self, mode="json"):
            return {"blocks": [], "lines": []}

    monkeypatch.setattr(runner_module, "run_model", lambda **kwargs: FakeResponse())
    out_dir = str(tmp_path / "out")
    code = nmk.main(
        [
            "--task",
            "segment",
            "--model",
            "ppocr",
            "--job-id",
            "job-1",
            "--project-id",
            "proj-1",
            "--document-id",
            "doc-1",
            "--part-id",
            "part-1",
            "--out",
            out_dir,
        ],
        transport=transport,
    )
    assert code == 0
    report_json = Path(out_dir, "report.json").read_text(encoding="utf-8")
    report_md = Path(out_dir, "report.md").read_text(encoding="utf-8")
    assert secret not in report_json
    assert secret not in report_md
    assert "[REDACTED]" in report_json
    assert os.environ["NOMIKOS_TOKEN"] == secret
