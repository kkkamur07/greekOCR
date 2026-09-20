#!/usr/bin/env python3
"""Prove the medium-det ONNX reproduces Paddle static inference on 14 pages.

ADR 0006 serves models through onnxruntime on CPU, so this script is the
measurement the export stands on. It runs as one stage at a time, each stage
writing its own JSON under the workdir so stages can be run and re-run alone:

* ``maps``: build the input tensor with PaddleX 3.7.0's own preprocessing
  classes, run the SAME tensor through the Paddle static predictor (mkldnn)
  and onnxruntime, compare the raw probability maps, and run PaddleX's own DB
  postprocess on both maps to compare the boxes. All 14 pages, caps 1920 and
  960.
* ``shapes``: five extra input sizes on synthetic tensors, so a frozen axis
  fails loudly instead of passing silently.
* ``pipeline``: run the full ``paddleocr.TextDetection`` pipeline from the raw
  image to prove the hand-built tensor path is faithful, and save the
  reference fixtures for the serving adapter.
* ``timing``: detector plus postprocess per page at cap 1920 on 3 pages, at 8
  threads and at 2 threads.
* ``report``: merge the stage JSON files, apply the pass gate, write the
  report, and exit non-zero on failure.

Every stage except ``timing`` runs both runtimes at 4 threads, so production
keeps half the box. Thread count does not change the numerics measured here.

Example (one line per stage; each appends to its own log file)::

    BOX=/root/ppocrv6-onnx-parity-20260919
    LIBS=/root/ppocrv6-bench-20260919/system-libs/sysroot/usr/lib/x86_64-linux-gnu
    export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 LD_LIBRARY_PATH=$LIBS
    cd $BOX
    ./venv/bin/python verify_parity.py --model-dir ... --onnx ... \\
        --images /root/orli-bench/images \\
        --grec-images /root/ppocrv6-bench-20260919/images \\
        --workdir $BOX/run maps > run/maps.log 2>&1
    ./venv/bin/python verify_parity.py --model-dir ... --onnx ... \\
        --images /root/orli-bench/images \\
        --grec-images /root/ppocrv6-bench-20260919/images \\
        --workdir $BOX/run shapes > run/shapes.log 2>&1
    ./venv/bin/python verify_parity.py --model-dir ... --onnx ... \\
        --images /root/orli-bench/images \\
        --grec-images /root/ppocrv6-bench-20260919/images \\
        --workdir $BOX/run pipeline > run/pipeline.log 2>&1
    ./venv/bin/python verify_parity.py --model-dir ... --onnx ... \\
        --images /root/orli-bench/images \\
        --grec-images /root/ppocrv6-bench-20260919/images \\
        --workdir $BOX/run timing > run/timing.log 2>&1
    ./venv/bin/python verify_parity.py --model-dir ... --onnx ... \\
        --images /root/orli-bench/images \\
        --grec-images /root/ppocrv6-bench-20260919/images \\
        --workdir $BOX/run report \\
        --report-json $BOX/parity-report.json > run/report.log 2>&1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np

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
    "grec-p1",
    "grec-p4",
)
CAPS = (1920, 960)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")
MAP_PAGES = ("c13", "grec-p1")
TIMING_PAGES = ("c13", "vat-1r", "grec-p1")
# (label, height, width): small, very wide, tall, and two odd multiples of 32.
SHAPE_CASES = (
    ("320x320", 320, 320),
    ("wide-1920x320", 320, 1920),
    ("tall-640x1920", 1920, 640),
    ("odd-480x736", 736, 480),
    ("odd-1056x864", 864, 1056),
)
# Stages other than timing share the box with production at 4 threads.
STAGE_THREADS = 4


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="PaddleX model directory")
    parser.add_argument("--onnx", type=Path, required=True, help="Exported ONNX file")
    parser.add_argument("--images", type=Path, required=True, help="Directory of manuscript pages")
    parser.add_argument("--grec-images", type=Path, required=True, help="Directory with grec pages")
    parser.add_argument(
        "--workdir", type=Path, required=True, help="Scratch dir for stage JSON and maps"
    )
    parser.add_argument("--report-json", type=Path, help="Where the report stage writes the report")
    parser.add_argument(
        "stage", choices=("maps", "shapes", "pipeline", "timing", "report"), help="Stage to run"
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_image(name: str, images: Path, grec_images: Path) -> Path:
    for directory in (images, grec_images):
        for suffix in IMAGE_SUFFIXES:
            candidate = directory / f"{name}{suffix}"
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"no image for page {name} under {images} or {grec_images}")


def _read_prepost(model_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    import yaml

    config = yaml.safe_load((model_dir / "inference.yml").read_text())
    pre_ops: dict[str, object] = {}
    for entry in config["PreProcess"]["transform_ops"]:
        key = next(iter(entry))
        pre_ops[key] = entry[key] or {}
    return pre_ops, config["PostProcess"]


def _norm_kwargs(model_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    pre_ops, post_cfg = _read_prepost(model_dir)
    norm_entry = pre_ops["NormalizeImage"]
    if not isinstance(norm_entry, dict):
        raise ValueError(f"unexpected NormalizeImage config: {norm_entry!r}")
    return (
        {
            "scale": norm_entry.get("scale", 1.0 / 255.0),
            "mean": norm_entry.get("mean", [0.485, 0.456, 0.406]),
            "std": norm_entry.get("std", [0.229, 0.224, 0.225]),
            "order": norm_entry.get("order", "hwc"),
        },
        post_cfg,
    )


def _build_tensor(
    bgr: np.ndarray, *, cap: int, norm_kwargs: dict[str, object]
) -> tuple[np.ndarray, np.ndarray]:
    """Run PaddleX's own preprocessing classes over one BGR image."""
    from paddlex.inference.models.common import ToBatch, ToCHWImage
    from paddlex.inference.models.text_detection.processors import (
        DetResizeForTest,
        NormalizeImage,
    )

    resizer = DetResizeForTest(limit_side_len=cap, limit_type="max", input_shape=None)
    resized, shapes = resizer(imgs=[bgr])
    normalizer = NormalizeImage(
        scale=norm_kwargs["scale"],
        mean=norm_kwargs["mean"],
        std=norm_kwargs["std"],
        order=norm_kwargs["order"],
    )
    normalized = normalizer(imgs=resized)
    chw = ToCHWImage()(imgs=normalized)
    batch = ToBatch()(imgs=chw)[0]
    return batch.astype(np.float32, copy=False), np.asarray(shapes[0])


def _make_post_op(post_cfg: dict[str, object], thresh: float, box_thresh: float):
    from paddlex.inference.models.text_detection.processors import DBPostProcess

    if post_cfg.get("name") != "DBPostProcess":
        raise ValueError(f"unexpected postprocess: {post_cfg.get('name')}")
    return DBPostProcess(
        thresh=thresh,
        box_thresh=box_thresh,
        max_candidates=int(post_cfg.get("max_candidates", 1000)),
        unclip_ratio=float(post_cfg.get("unclip_ratio", 1.4)),
        use_dilation=bool(post_cfg.get("use_dilation", False)),
        score_mode=str(post_cfg.get("score_mode", "fast")),
        box_type=str(post_cfg.get("box_type", "quad")),
    )


def _make_paddle_predictor(model_dir: Path, threads: int):
    """Mirror the PaddleX CPU recipe used by the reference benchmark."""
    from paddle import inference as paddle_inference

    config = paddle_inference.Config(
        str(model_dir / "inference.json"), str(model_dir / "inference.pdiparams")
    )
    config.disable_gpu()
    config.enable_mkldnn()
    config.set_mkldnn_cache_capacity(10)
    config.set_cpu_math_library_num_threads(threads)
    if hasattr(config, "enable_new_ir"):
        config.enable_new_ir(True)
    if hasattr(config, "enable_new_executor"):
        config.enable_new_executor()
    config.set_optimization_level(3)
    config.enable_memory_optim()
    config.disable_glog_info()
    return paddle_inference.create_predictor(config)


def _run_paddle(predictor, tensor: np.ndarray) -> np.ndarray:
    input_names = predictor.get_input_names()
    handle = predictor.get_input_handle(input_names[0])
    handle.reshape(list(tensor.shape))
    handle.copy_from_cpu(tensor)
    predictor.run()
    output_names = predictor.get_output_names()
    return np.asarray(predictor.get_output_handle(output_names[0]).copy_to_cpu())


def _make_ort_session(onnx_path: Path, threads: int):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def _run_onnx(session, tensor: np.ndarray) -> np.ndarray:
    name = session.get_inputs()[0].name
    return np.asarray(session.run(None, {name: tensor})[0])


def _as_boxes(raw: object) -> np.ndarray:
    array = np.asarray(raw, dtype=np.float64)
    if array.size == 0:
        return np.zeros((0, 4, 2), dtype=np.float64)
    return array.reshape(-1, 4, 2)


def _max_corner_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Greedy-bijection max corner distance between two equal-size box sets."""
    if len(a) == 0 and len(b) == 0:
        return 0.0
    remaining_b = list(range(len(b)))
    worst = 0.0
    for box_a in a:
        best_j, best_cost = -1, float("inf")
        for j in remaining_b:
            cost = float(np.linalg.norm(box_a - b[j], axis=1).max())
            if cost < best_cost:
                best_j, best_cost = j, cost
        remaining_b.remove(best_j)
        worst = max(worst, best_cost)
    return worst


def _versions() -> dict[str, str]:
    import importlib.metadata as metadata

    return {
        name: metadata.version(name)
        for name in ("paddlepaddle", "paddleocr", "paddlex", "onnxruntime", "numpy")
    }


def cmd_maps(args: argparse.Namespace) -> int:
    import cv2

    cv2.setNumThreads(1)
    norm_kwargs, post_cfg = _norm_kwargs(args.model_dir)
    thresh = float(post_cfg.get("thresh", 0.2))
    box_thresh = float(post_cfg.get("box_thresh", 0.45))
    post_op = _make_post_op(post_cfg, thresh, box_thresh)
    args.workdir.mkdir(parents=True, exist_ok=True)
    maps_dir = args.workdir / "maps"
    maps_dir.mkdir(exist_ok=True)

    predictor = _make_paddle_predictor(args.model_dir, STAGE_THREADS)
    session = _make_ort_session(args.onnx, STAGE_THREADS)
    per_page: list[dict[str, object]] = []
    failures: list[str] = []
    for page in PAGES:
        image_path = _resolve_image(page, args.images, args.grec_images)
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"could not decode {image_path}")
        for cap in CAPS:
            tensor, shape = _build_tensor(bgr, cap=cap, norm_kwargs=norm_kwargs)
            paddle_map = _run_paddle(predictor, tensor).astype(np.float64)
            onnx_map = _run_onnx(session, tensor).astype(np.float64)
            paddle_prob = paddle_map.reshape(paddle_map.shape[-2], paddle_map.shape[-1])
            onnx_prob = onnx_map.reshape(onnx_map.shape[-2], onnx_map.shape[-1])
            abs_diff = np.abs(paddle_prob - onnx_prob)
            max_abs = float(abs_diff.max())
            mean_abs = float(abs_diff.mean())
            bin_diff = int(np.count_nonzero((paddle_prob > thresh) != (onnx_prob > thresh)))

            paddle_boxes_raw, _ = post_op([paddle_map.astype(np.float32)], [shape])
            onnx_boxes_raw, _ = post_op([onnx_map.astype(np.float32)], [shape])
            paddle_boxes = _as_boxes(paddle_boxes_raw)
            onnx_boxes = _as_boxes(onnx_boxes_raw)
            box_delta = int(len(paddle_boxes) - len(onnx_boxes))
            corner = (
                _max_corner_distance(paddle_boxes, onnx_boxes) if box_delta == 0 else float("inf")
            )
            per_page.append(
                {
                    "page": page,
                    "cap": cap,
                    "tensor_shape": [int(v) for v in tensor.shape],
                    "paddle_boxes": int(len(paddle_boxes)),
                    "onnx_boxes": int(len(onnx_boxes)),
                    "box_delta": box_delta,
                    "max_corner_px": corner,
                    "max_abs_prob_diff": max_abs,
                    "mean_abs_prob_diff": mean_abs,
                    "binarisation_diff_px": bin_diff,
                }
            )
            print(
                f"{page} cap={cap} boxes paddle={len(paddle_boxes)} onnx={len(onnx_boxes)} "
                f"delta={box_delta} corner={corner:.3f}px maxabs={max_abs:.2e} "
                f"meanabs={mean_abs:.2e} bindiff={bin_diff}",
                flush=True,
            )
            if box_delta != 0:
                failures.append(f"{page} cap={cap}: box count delta {box_delta}")
            elif corner > 1.0:
                failures.append(f"{page} cap={cap}: corner distance {corner:.3f}px")
            if max_abs > 1e-2:
                failures.append(f"{page} cap={cap}: max abs prob diff {max_abs:.2e}")
            if cap == 1920 and page in MAP_PAGES:
                np.save(maps_dir / f"{page}.paddle.npy", paddle_map.astype(np.float32))
                np.save(maps_dir / f"{page}.onnx.npy", onnx_map.astype(np.float32))
    (args.workdir / "maps.json").write_text(
        json.dumps({"per_page": per_page, "failures": failures}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"maps done: failures={failures}", flush=True)
    return 0 if not failures else 1


def cmd_shapes(args: argparse.Namespace) -> int:
    predictor = _make_paddle_predictor(args.model_dir, STAGE_THREADS)
    session = _make_ort_session(args.onnx, STAGE_THREADS)
    rng = np.random.default_rng(0)
    shape_rows: list[dict[str, object]] = []
    failures: list[str] = []
    for label, height, width in SHAPE_CASES:
        tensor = rng.normal(0.0, 1.0, size=(1, 3, height, width)).astype(np.float32)
        paddle_map = _run_paddle(predictor, tensor).astype(np.float64)
        onnx_map = _run_onnx(session, tensor).astype(np.float64)
        max_abs = float(np.abs(paddle_map - onnx_map).max())
        ok_shape = tuple(int(v) for v in onnx_map.shape) == (1, 1, height, width)
        shape_rows.append(
            {
                "label": label,
                "input": [1, 3, height, width],
                "max_abs_prob_diff": max_abs,
                "onnx_shape_ok": ok_shape,
            }
        )
        print(f"shape {label}: maxabs={max_abs:.2e} shape_ok={ok_shape}", flush=True)
        if not ok_shape:
            failures.append(f"shape {label}: onnx output shape {tuple(onnx_map.shape)}")
        if max_abs > 1e-2:
            failures.append(f"shape {label}: max abs prob diff {max_abs:.2e}")
    (args.workdir / "shapes.json").write_text(
        json.dumps({"shapes": shape_rows, "failures": failures}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"shapes done: failures={failures}", flush=True)
    return 0 if not failures else 1


def cmd_pipeline(args: argparse.Namespace) -> int:
    import cv2

    cv2.setNumThreads(1)
    from paddleocr import TextDetection

    norm_kwargs, post_cfg = _norm_kwargs(args.model_dir)
    thresh = float(post_cfg.get("thresh", 0.2))
    box_thresh = float(post_cfg.get("box_thresh", 0.45))
    post_op = _make_post_op(post_cfg, thresh, box_thresh)
    fixtures_dir = args.workdir / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    predictor = _make_paddle_predictor(args.model_dir, STAGE_THREADS)
    pipeline = TextDetection(
        model_name="PP-OCRv6_medium_det",
        model_dir=str(args.model_dir),
        device="cpu",
        engine="paddle_static",
        engine_config={
            "run_mode": "mkldnn",
            "cpu_threads": STAGE_THREADS,
            "mkldnn_cache_capacity": 10,
            "enable_cinn": False,
        },
        enable_hpi=False,
        limit_type="max",
        thresh=thresh,
        box_thresh=box_thresh,
        unclip_ratio=float(post_cfg.get("unclip_ratio", 1.4)),
    )
    pipeline_rows: list[dict[str, object]] = []
    failures: list[str] = []
    for page in PAGES:
        image_path = _resolve_image(page, args.images, args.grec_images)
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"could not decode {image_path}")
        for cap in CAPS:
            results = pipeline.predict(bgr, batch_size=1, limit_side_len=cap, limit_type="max")
            if len(results) != 1:
                raise RuntimeError(f"expected one result for {page} cap={cap}")
            pipe_boxes = _as_boxes(np.asarray(results[0]["dt_polys"], dtype=np.float64))
            pipe_scores = np.asarray(results[0]["dt_scores"], dtype=np.float64)
            tensor, shape = _build_tensor(bgr, cap=cap, norm_kwargs=norm_kwargs)
            hand_map = _run_paddle(predictor, tensor)
            hand_boxes_raw, _ = post_op([hand_map.astype(np.float32)], [shape])
            hand_boxes = _as_boxes(hand_boxes_raw)
            delta = int(len(hand_boxes) - len(pipe_boxes))
            corner = _max_corner_distance(hand_boxes, pipe_boxes) if delta == 0 else float("inf")
            pipeline_rows.append(
                {
                    "page": page,
                    "cap": cap,
                    "pipeline_boxes": int(len(pipe_boxes)),
                    "hand_delta": delta,
                    "hand_corner_px": corner,
                }
            )
            print(
                f"pipeline {page} cap={cap}: boxes={len(pipe_boxes)} "
                f"hand_delta={delta} hand_corner={corner:.3f}px",
                flush=True,
            )
            if delta != 0 or corner > 1.0:
                failures.append(f"{page} cap={cap}: hand tensor path differs from pipeline")
            if cap == 1920:
                fixture = {
                    "image": image_path.name,
                    "image_sha256": _sha256(image_path),
                    "original_height": int(bgr.shape[0]),
                    "original_width": int(bgr.shape[1]),
                    "tensor_shape": [int(v) for v in tensor.shape],
                    "boxes": pipe_boxes.tolist(),
                    "scores": [float(v) for v in pipe_scores.tolist()],
                }
                (fixtures_dir / f"{page}.json").write_text(
                    json.dumps(fixture, indent=2) + "\n", encoding="utf-8"
                )
    pipeline.close()
    (args.workdir / "pipeline.json").write_text(
        json.dumps({"pipeline_check": pipeline_rows, "failures": failures}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"pipeline done: failures={failures}", flush=True)
    return 0 if not failures else 1


def cmd_timing(args: argparse.Namespace) -> int:
    import cv2

    cv2.setNumThreads(1)
    norm_kwargs, post_cfg = _norm_kwargs(args.model_dir)
    thresh = float(post_cfg.get("thresh", 0.2))
    box_thresh = float(post_cfg.get("box_thresh", 0.45))
    post_op = _make_post_op(post_cfg, thresh, box_thresh)
    timing_rows: list[dict[str, object]] = []
    for threads in (8, 2):
        predictor = _make_paddle_predictor(args.model_dir, threads)
        session = _make_ort_session(args.onnx, threads)
        for page in TIMING_PAGES:
            image_path = _resolve_image(page, args.images, args.grec_images)
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"could not decode {image_path}")
            tensor, shape = _build_tensor(bgr, cap=1920, norm_kwargs=norm_kwargs)
            paddle_samples, onnx_samples = [], []
            for repeat in range(4):
                started = time.perf_counter()
                paddle_map = _run_paddle(predictor, tensor)
                post_op([paddle_map.astype(np.float32)], [shape])
                elapsed_paddle = time.perf_counter() - started
                started = time.perf_counter()
                onnx_map = _run_onnx(session, tensor)
                post_op([onnx_map.astype(np.float32)], [shape])
                elapsed_onnx = time.perf_counter() - started
                if repeat >= 1:
                    paddle_samples.append(elapsed_paddle)
                    onnx_samples.append(elapsed_onnx)
            row = {
                "page": page,
                "threads": threads,
                "paddle_median_s": float(statistics.median(paddle_samples)),
                "onnx_median_s": float(statistics.median(onnx_samples)),
            }
            timing_rows.append(row)
            print(
                f"timing {page} threads={threads}: "
                f"paddle={row['paddle_median_s']:.3f}s onnx={row['onnx_median_s']:.3f}s",
                flush=True,
            )
        del predictor, session
    (args.workdir / "timing.json").write_text(
        json.dumps({"timing": timing_rows, "failures": []}, indent=2) + "\n", encoding="utf-8"
    )
    print("timing done", flush=True)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    if args.report_json is None:
        raise ValueError("report stage needs --report-json")
    maps = json.loads((args.workdir / "maps.json").read_text())
    shapes = json.loads((args.workdir / "shapes.json").read_text())
    pipeline = json.loads((args.workdir / "pipeline.json").read_text())
    timing = json.loads((args.workdir / "timing.json").read_text())
    failures: list[str] = (
        list(maps["failures"])
        + list(shapes["failures"])
        + list(pipeline["failures"])
        + list(timing["failures"])
    )
    report = {
        "versions": _versions(),
        "model_dir": str(args.model_dir),
        "onnx": str(args.onnx),
        "per_page": maps["per_page"],
        "shapes": shapes["shapes"],
        "timing": timing["timing"],
        "pipeline_check": pipeline["pipeline_check"],
        "failures": failures,
        "passed": not failures,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in maps["per_page"]:
        print(
            f"{row['page']} cap={row['cap']} boxes "
            f"paddle={row['paddle_boxes']} onnx={row['onnx_boxes']} "
            f"delta={row['box_delta']} corner={row['max_corner_px']:.3f}px "
            f"maxabs={row['max_abs_prob_diff']:.2e} "
            f"meanabs={row['mean_abs_prob_diff']:.2e} "
            f"bindiff={row['binarisation_diff_px']}",
            flush=True,
        )
    for row in shapes["shapes"]:
        print(
            f"shape {row['label']}: maxabs={row['max_abs_prob_diff']:.2e} "
            f"shape_ok={row['onnx_shape_ok']}",
            flush=True,
        )
    for row in timing["timing"]:
        print(
            f"timing {row['page']} threads={row['threads']}: "
            f"paddle={row['paddle_median_s']:.3f}s onnx={row['onnx_median_s']:.3f}s",
            flush=True,
        )
    for row in pipeline["pipeline_check"]:
        print(
            f"pipeline {row['page']} cap={row['cap']}: boxes={row['pipeline_boxes']} "
            f"hand_delta={row['hand_delta']} hand_corner={row['hand_corner_px']:.3f}px",
            flush=True,
        )
    print(f"passed={report['passed']} failures={failures}", flush=True)
    return 0 if report["passed"] else 1


def main() -> int:
    args = _parse_args()
    if args.stage == "maps":
        return cmd_maps(args)
    if args.stage == "shapes":
        return cmd_shapes(args)
    if args.stage == "pipeline":
        return cmd_pipeline(args)
    if args.stage == "timing":
        return cmd_timing(args)
    return cmd_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
