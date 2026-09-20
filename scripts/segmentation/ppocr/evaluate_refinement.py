#!/usr/bin/env python3
"""Score the ppocr-det adapter with refinement off, default and drop policies.

Runs the production entry point on the 12 Coptic benchmark pages three ways
and scores each response against the same target lines with the benchmark's
own line matching: a prediction matches a target when at least 50 percent of
31 mid-axis samples fall inside the target polygon with at least 50 percent
horizontal overlap, assigned one to one for maximum cardinality then affinity
(server/ppocrv6_quality_eval_20260919.py: `axis`, `assign`, `finish`). The
benchmark's own tool recovers 1008 of 1018 care targets on these pages; this
port recovers 1002, so it is a consistent off-vs-defaults comparator rather
than identical tooling.

Prints per variant: detections, precision, recall, F1, care targets covered
by exactly one detection, line pairs sharing at least 20 percent area,
suspects flagged and real lines wrongly flagged. Also writes reading-order
overlays for vat-1r and c13 with suspects in red. The pass gate lives in the
release report, not in the exit code: this script always exits 0.

Example::

    PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 /Users/krishuagarwal/Desktop/Programming/python/greekOCR/.venv/bin/python \\
        scripts/segmentation/ppocr/evaluate_refinement.py \\
        --fixtures /Users/krishuagarwal/Desktop/Programming/python/greekOCR-wt/_ppocr-parity/fixtures \\
        --images /Users/krishuagarwal/Desktop/Programming/python/greekOCR-wt/_orli-env/server/images \\
        --gt /Users/krishuagarwal/Desktop/Programming/python/greekOCR-wt/_gt-recon/human-guide/v4/recon \\
        --onnx /Users/krishuagarwal/Desktop/Programming/python/greekOCR-wt/_ppocr-parity/pp-ocrv6-medium-det.onnx \\
        --output-dir /Users/krishuagarwal/Desktop/Programming/python/greekOCR-wt/_ppocr-parity/refinement
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString, Point, Polygon

PAGES = (
    "vat-1r",
    "vat-1v",
    "vat-2r",
    "vat-2v",
    "vat-3r",
    "vat-7v",
    "c10",
    "c11",
    "c12",
    "c13",
    "c14",
    "c21",
)
OVERLAY_PAGES = ("vat-1r", "c13")
N_SAMPLES = 31
PAIR_THRESHOLD = 0.20


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures", type=Path, required=True, help="Directory of <page>.json fixtures"
    )
    parser.add_argument("--images", type=Path, required=True, help="Directory of page images")
    parser.add_argument("--gt", type=Path, required=True, help="Ground truth recon directory")
    parser.add_argument("--onnx", type=Path, required=True, help="Adapter ONNX artifact")
    parser.add_argument("--artifact-sha256", default=None, help="Expected SHA-256 of the ONNX")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where overlays go")
    parser.add_argument(
        "--merge-gap-ratio",
        action="append",
        default=[],
        help=(
            "Score the defaults and drop variants at these merge gap ratios "
            "(repeatable or comma separated). In sweep mode overlays are "
            "skipped and the JSON gains a per-ratio sweep summary."
        ),
    )
    return parser.parse_args()


def _sweep_ratios(raw: list[str]) -> list[float]:
    ratios = []
    for chunk in raw:
        ratios.extend(float(part) for part in chunk.split(",") if part.strip())
    if not ratios:
        raise ValueError("--merge-gap-ratio needs at least one number")
    return ratios


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_image(page: str, fixtures: Path, images: Path) -> bytes:
    fixture = json.loads((fixtures / f"{page}.json").read_text(encoding="utf-8"))
    data = (images / fixture["image"]).read_bytes()
    if _sha256(data) != fixture["image_sha256"]:
        raise ValueError(f"image SHA-256 does not match fixture {page}.json")
    return data


def _load_targets(gt_dir: Path, page: str) -> list[dict]:
    """Care and ignore target polygons, as in the benchmark's capture_inputs."""
    doc = json.loads((gt_dir / f"{page}-latest.json").read_text(encoding="utf-8"))
    paired = {
        text["paired_line_id"]
        for text in doc["pairing"]["text_lines"]
        if text.get("paired_line_id")
    }
    targets = []
    for line in doc["lines"]:
        polygon = Polygon(line["points"])
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        targets.append({"id": line["id"], "polygon": polygon, "care": line["id"] in paired})
    return targets


def _axis(polygon: Polygon) -> list[list[float]]:
    """31 mid-height axis samples of a quad, per the benchmark's `axis`."""
    x0, y0, x1, y1 = polygon.bounds
    if x1 <= x0:
        return []
    points = []
    for fraction in (np.arange(N_SAMPLES) + 0.5) / N_SAMPLES:
        x = float(x0 + fraction * (x1 - x0))
        section = polygon.intersection(LineString([(x, y0 - 1), (x, y1 + 1)]))
        if section.is_empty:
            return []
        points.append([x, (section.bounds[1] + section.bounds[3]) / 2])
    return points


def _score_page(targets: list[dict], polygons: list[Polygon]) -> dict:
    """One-to-one line matching, per the benchmark's `score` and `finish`."""
    care = [i for i, target in enumerate(targets) if target["care"]]
    ignore = [i for i, target in enumerate(targets) if not target["care"]]
    eligible = np.zeros((len(polygons), len(targets)), dtype=bool)
    affinity = np.zeros(eligible.shape, dtype=float)
    for pi, polygon in enumerate(polygons):
        samples = _axis(polygon)
        if not samples:
            continue
        extent = (polygon.bounds[0], polygon.bounds[2])
        for gi, target in enumerate(targets):
            gt = target["polygon"]
            x0, _, x1, _ = gt.bounds
            overlap = max(0.0, min(extent[1], x1) - max(extent[0], x0)) / max(x1 - x0, 1e-12)
            if overlap < 0.5:
                continue
            inside = sum(gt.covers(Point(point)) for point in samples) / N_SAMPLES
            eligible[pi, gi] = inside >= 0.5
            affinity[pi, gi] = (inside + min(overlap, 1.0)) / 2
    weight = eligible[:, care] * (min(eligible[:, care].shape) + 1 + affinity[:, care])
    paired: list[tuple[int, int]] = []
    used: set[int] = set()
    if weight.size:
        rows, cols = linear_sum_assignment(weight, maximize=True)
        for row, col in zip(rows, cols, strict=True):
            if eligible[int(row), care[int(col)]]:
                paired.append((int(row), care[int(col)]))
                used.add(int(row))
    ignored = [
        p
        for p in range(len(polygons))
        if p not in used
        and any(eligible[p, g] for g in ignore)
        and not any(eligible[p, g] for g in care)
    ]
    fps = sorted(set(range(len(polygons))) - used - set(ignored))
    fns = sorted(set(care) - {g for _, g in paired})
    alone = sum(1 for _, g in paired if sum(1 for p in range(len(polygons)) if eligible[p, g]) == 1)
    paired_ids = sorted(targets[g]["id"] for _, g in paired)
    alone_ids = sorted(
        targets[g]["id"]
        for _, g in paired
        if sum(1 for p in range(len(polygons)) if eligible[p, g]) == 1
    )
    return {
        "paired_ids": paired_ids,
        "alone_ids": alone_ids,
        "n_gt": len(care),
        "n_pred": len(polygons),
        "tp": len(paired),
        "fp": len(fps),
        "fn": len(fns),
        "ignored": len(ignored),
        "alone": alone,
        "matched_pred": sorted(used),
    }


def _pairs_above(polygons: list[Polygon], threshold: float) -> int:
    """Line pairs sharing at least `threshold` of the smaller polygon."""
    count = 0
    for left, right in itertools.combinations(polygons, 2):
        smaller = min(left.area, right.area)
        if smaller > 0 and left.intersection(right).area / smaller >= threshold:
            count += 1
    return count


def _draw_overlay(page: str, image_bytes: bytes, lines: list, output_dir: Path) -> list[int]:
    """Reading-order overlay: body quads green with numbers, suspects red.

    Drawing mirrors verify_adapter.py (green quads, order numbers beside the
    left edge, yellow baselines); suspects use red for both quad and number.
    Returns the suspect line numbers.
    """
    raw = np.frombuffer(image_bytes, dtype=np.uint8)
    canvas = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if canvas is None:
        raise ValueError(f"could not decode image for page {page}")
    height, width = canvas.shape[:2]
    scale = max(width, height) / 1400.0
    thickness = max(1, int(round(2 * scale)))
    font_scale = max(0.7, 1.1 * scale)
    font_thickness = max(1, int(round(2 * scale)))

    suspect_numbers = []
    for position, line in enumerate(lines):
        number = position + 1
        quad = np.asarray(line["points"], dtype=np.float64)
        suspect = bool(line["source_metadata"].get("suspect", False))
        colour = (0, 0, 255) if suspect else (0, 200, 0)
        if suspect:
            suspect_numbers.append(number)
        cv2.polylines(
            canvas, [quad.astype(np.int32).reshape(-1, 1, 2)], True, colour, thickness, cv2.LINE_AA
        )
        baseline = np.asarray(line["baseline"]["points"], dtype=np.float64)
        if not suspect:
            cv2.line(
                canvas,
                (int(baseline[0][0]), int(baseline[0][1])),
                (int(baseline[1][0]), int(baseline[1][1])),
                (0, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
        left_x = float(quad[:, 0].min())
        mid_y = float(quad[:, 1].mean())
        label = str(number)
        (label_w, label_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        text_x = int(min(max(left_x - label_w - 6 * scale, 0), width - label_w - 1))
        text_y = int(min(max(mid_y + label_h / 2, label_h), height - 1))
        cv2.putText(
            canvas,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            font_thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(
        str(output_dir / f"{page}.refined.overlay.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90]
    )
    if not ok:
        raise RuntimeError(f"could not write overlay for page {page}")
    return suspect_numbers


def main() -> int:
    from nomikos_inference.architectures.ppocr_det import run_ppocr_det_segment

    args = _parse_args()
    sweep = _sweep_ratios(args.merge_gap_ratio) if args.merge_gap_ratio else []
    if sweep:
        variants: dict[str, dict | None] = {
            "off": {
                "merge_fragments": False,
                "resolve_overlaps": False,
                "noise_policy": "off",
            },
        }
        for ratio in sweep:
            variants[f"gap-{ratio}"] = {"merge_gap_ratio": ratio}
            variants[f"drop-{ratio}"] = {"noise_policy": "drop", "merge_gap_ratio": ratio}
    else:
        variants = {
            "off": {
                "merge_fragments": False,
                "resolve_overlaps": False,
                "noise_policy": "off",
            },
            "defaults": None,
            "drop": {"noise_policy": "drop"},
        }
    totals = {
        name: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "n_pred": 0,
            "alone": 0,
            "pairs": 0,
            "suspects": 0,
            "wrong": 0,
            "merged": 0,
        }
        for name in variants
    }
    pages: dict[str, dict] = {}
    print(
        "variant detections precision recall f1 alone pairs suspects wrong",
        flush=True,
    )
    for page in PAGES:
        image_bytes = _load_image(page, args.fixtures, args.images)
        targets = _load_targets(args.gt, page)
        pages[page] = {}
        for name, params in variants.items():
            response = run_ppocr_det_segment(
                image_bytes,
                model_path=args.onnx,
                artifact_sha256=args.artifact_sha256,
                params=params,
            )
            polygons = [Polygon(line.points) for line in response.lines]
            scored = _score_page(targets, polygons)
            suspects = [
                position
                for position, line in enumerate(response.lines)
                if line.source_metadata.get("suspect", False)
            ]
            wrong = sum(1 for p in suspects if p in scored["matched_pred"])
            merged = sum(
                1 for line in response.lines if line.source_metadata.get("merged_from", 1) > 1
            )
            pages[page][name] = {
                **scored,
                "pairs": _pairs_above(polygons, PAIR_THRESHOLD),
                "suspects": len(suspects),
                "wrong": wrong,
                "merged": merged,
            }
            for key in ("tp", "fp", "fn", "n_pred", "alone"):
                totals[name][key] += scored[key]
            totals[name]["pairs"] += pages[page][name]["pairs"]
            totals[name]["suspects"] += len(suspects)
            totals[name]["wrong"] += wrong
            totals[name]["merged"] += merged
        if not sweep and page in OVERLAY_PAGES:
            defaults = run_ppocr_det_segment(
                image_bytes, model_path=args.onnx, artifact_sha256=args.artifact_sha256, params=None
            )
            serial = [
                {
                    "points": line.points,
                    "baseline": line.baseline,
                    "source_metadata": line.source_metadata,
                }
                for line in defaults.lines
            ]
            numbers = _draw_overlay(page, image_bytes, serial, args.output_dir)
            pages[page]["suspect_numbers"] = numbers
            print(f"{page} suspect line numbers: {numbers}", flush=True)
    for name in variants:
        total = totals[name]
        precision = total["tp"] / (total["tp"] + total["fp"]) if total["tp"] + total["fp"] else 0.0
        recall = total["tp"] / (total["tp"] + total["fn"]) if total["tp"] + total["fn"] else 0.0
        f1 = (
            2 * total["tp"] / (2 * total["tp"] + total["fp"] + total["fn"])
            if 2 * total["tp"] + total["fp"] + total["fn"]
            else 0.0
        )
        total["precision"] = precision
        total["recall"] = recall
        total["f1"] = f1
        print(
            f"{name} detections={total['n_pred']} P={precision:.4f} R={recall:.4f} "
            f"F1={f1:.4f} alone={total['alone']} pairs={total['pairs']} "
            f"suspects={total['suspects']} wrong={total['wrong']} merged={total['merged']}",
            flush=True,
        )
    payload: dict = {"pages": pages, "totals": totals}
    if sweep:
        payload["sweep"] = _summarise_sweep(pages, totals, sweep)
    (args.output_dir / "refinement-evaluation.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return 0


def _summarise_sweep(pages: dict, totals: dict, sweep: list[float]) -> dict:
    """Per-ratio summary: metrics, lines lost against off, lines fixed, merges."""
    summary = {}
    print("ratio R P Pdrop F1 F1drop lost fixed merged", flush=True)
    for ratio in sweep:
        gap, drop = totals[f"gap-{ratio}"], totals[f"drop-{ratio}"]
        lost: dict[str, list] = {}
        fixed: dict[str, list] = {}
        for page, entry in pages.items():
            off_paired = set(entry["off"]["paired_ids"])
            off_alone = set(entry["off"]["alone_ids"])
            gap_paired = set(entry[f"gap-{ratio}"]["paired_ids"])
            gap_alone = set(entry[f"gap-{ratio}"]["alone_ids"])
            page_lost = sorted(off_paired - gap_paired)
            page_fixed = sorted(gap_alone - off_alone)
            if page_lost:
                lost[page] = page_lost
            if page_fixed:
                fixed[page] = page_fixed
        summary[str(ratio)] = {
            "recall": gap["recall"],
            "precision": gap["precision"],
            "precision_drop": drop["precision"],
            "f1": gap["f1"],
            "f1_drop": drop["f1"],
            "lost": lost,
            "fixed": fixed,
            "merged": gap["merged"],
        }
        print(
            f"{ratio} R={gap['recall']:.4f} P={gap['precision']:.4f} "
            f"Pdrop={drop['precision']:.4f} F1={gap['f1']:.4f} "
            f"F1drop={drop['f1']:.4f} lost={sum(len(v) for v in lost.values())} "
            f"fixed={sum(len(v) for v in fixed.values())} merged={gap['merged']}",
            flush=True,
        )
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
