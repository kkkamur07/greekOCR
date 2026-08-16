"""Run Kraken BLLA segmentation on manuscript pages and compare with Transkribus PAGE XML."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PAGE_NS = {"p": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}


def parse_coords(points_str: str) -> list[tuple[int, int]]:
    return [(int(x), int(y)) for x, y in (p.split(",") for p in points_str.strip().split())]


def parse_transkribus_page(xml_path: Path) -> dict:
    """Extract text regions, lines, and baselines from PAGE XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    page = root.find(".//p:Page", PAGE_NS)
    result = {
        "width": int(page.get("imageWidth")),
        "height": int(page.get("imageHeight")),
        "regions": [],
        "lines": [],
        "baselines": [],
    }

    for region in page.findall(".//p:TextRegion", PAGE_NS):
        coords_el = region.find("p:Coords", PAGE_NS)
        if coords_el is not None and coords_el.get("points"):
            result["regions"].append(parse_coords(coords_el.get("points")))

        for line in region.findall("p:TextLine", PAGE_NS):
            line_coords = line.find("p:Coords", PAGE_NS)
            if line_coords is not None and line_coords.get("points"):
                result["lines"].append(parse_coords(line_coords.get("points")))
            baseline = line.find("p:Baseline", PAGE_NS)
            if baseline is not None and baseline.get("points"):
                result["baselines"].append(parse_coords(baseline.get("points")))

    return result


def run_kraken_segmentation(image_path: Path) -> dict:
    """Run Kraken's default BLLA segmentation on an image."""
    from kraken.blla import segment
    from kraken.lib.vgsl import TorchVGSLModel

    model = TorchVGSLModel.load_model(str(
        Path(sys.modules["kraken"].__file__).parent / "blla.mlmodel"
    ))
    img = Image.open(image_path)
    seg_result = segment(img, model=model)

    result = {"regions": [], "lines": [], "baselines": []}
    for line in seg_result.lines:
        if hasattr(line, "boundary") and line.boundary:
            result["lines"].append([(int(x), int(y)) for x, y in line.boundary])
        if hasattr(line, "baseline") and line.baseline:
            result["baselines"].append([(int(x), int(y)) for x, y in line.baseline])
    for region_type, regions in seg_result.regions.items():
        for region in regions:
            if hasattr(region, "boundary") and region.boundary:
                result["regions"].append([(int(x), int(y)) for x, y in region.boundary])
    return result


def draw_segmentation(image: np.ndarray, seg_data: dict, label: str, color_baselines=(0, 255, 0), color_regions=(255, 0, 0), color_lines=(0, 0, 255)) -> np.ndarray:
    """Draw segmentation overlays on image."""
    overlay = image.copy()

    for region in seg_data["regions"]:
        pts = np.array(region, dtype=np.int32)
        cv2.polylines(overlay, [pts], True, color_regions, 2)

    for line in seg_data["lines"]:
        pts = np.array(line, dtype=np.int32)
        cv2.polylines(overlay, [pts], True, color_lines, 1)

    for baseline in seg_data["baselines"]:
        pts = np.array(baseline, dtype=np.int32)
        cv2.polylines(overlay, [pts], False, color_baselines, 2)

    cv2.putText(overlay, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
    cv2.putText(overlay, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return overlay


def main() -> None:
    manuscripts = [
        ("armenian/MS-P-331-ff24r-34r", "png"),
        ("armenian/Ms_P_172-CanonsDvin719-Partaw768-Dvin644-Karin_complete", "jpg"),
    ]

    output_dir = REPO_ROOT / "segmentation_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    num_pages = 3

    for manuscript, img_ext in manuscripts:
        ms_dir = REPO_ROOT / "data" / "raw" / manuscript
        page_dir = ms_dir / "page"
        if not page_dir.exists():
            print(f"Skipping {manuscript}: no page/ directory")
            continue

        xml_files = sorted(page_dir.glob("*.xml"))[:num_pages]
        for xml_file in xml_files:
            stem = xml_file.stem
            img_candidates = [ms_dir / f"{stem}.{img_ext}", ms_dir / f"{stem}.png", ms_dir / f"{stem}.jpg"]
            img_path = next((p for p in img_candidates if p.exists()), None)
            if img_path is None:
                print(f"  Skipping {stem}: no image found")
                continue

            print(f"Processing {manuscript}/{stem}...")
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"  Failed to read {img_path}")
                continue

            transkribus_data = parse_transkribus_page(xml_file)
            print(f"  Transkribus: {len(transkribus_data['regions'])} regions, {len(transkribus_data['baselines'])} baselines")

            kraken_data = run_kraken_segmentation(img_path)
            print(f"  Kraken:      {len(kraken_data['regions'])} regions, {len(kraken_data['baselines'])} baselines")

            img_transkribus = draw_segmentation(image, transkribus_data, "Transkribus")
            img_kraken = draw_segmentation(image, kraken_data, "Kraken")

            h = max(img_transkribus.shape[0], img_kraken.shape[0])
            w = img_transkribus.shape[1] + img_kraken.shape[1] + 10
            combined = np.ones((h, w, 3), dtype=np.uint8) * 255
            combined[:img_transkribus.shape[0], :img_transkribus.shape[1]] = img_transkribus
            combined[:img_kraken.shape[0], img_transkribus.shape[1] + 10:] = img_kraken

            ms_short = manuscript.split("/")[-1]
            out_path = output_dir / f"{ms_short}_{stem}.jpg"
            cv2.imwrite(str(out_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f"  Saved: {out_path}")

    print(f"\nAll comparisons saved to {output_dir}")


if __name__ == "__main__":
    main()
