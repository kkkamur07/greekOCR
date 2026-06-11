"""Kraken whole-page binarization."""

from pathlib import Path

from PIL import Image

from annote.schemas.annotation import Segment
from annote.services.kraken_binarize import binarize_image, binarize_page, clear_binarized_page
from annote.services.page_image import is_binarized_active, processed_page_path, resolve_working_page_image
from annote.services.segment_refinement import refine_kraken_segments
from tests.conftest import minimal_jpeg_bytes


def test_binarize_page_persists_processed_image(client, data_root, monkeypatch):
    stem = "folio"
    (data_root / "manuscripts" / "pages" / f"{stem}.jpg").write_bytes(minimal_jpeg_bytes())

    def fake_binarize(image: Image.Image) -> Image.Image:
        assert image.size == (100, 50)
        return Image.new("RGB", image.size, "black")

    monkeypatch.setattr("annote.services.kraken_binarize.binarize_image", fake_binarize)

    response = client.post(f"/pages/{stem}/binarize")
    assert response.status_code == 200
    body = response.json()
    assert body["binarized_at"] is not None

    processed = processed_page_path(data_root, stem)
    assert processed.is_file()
    assert is_binarized_active(data_root, stem)
    assert resolve_working_page_image(data_root, stem) == processed


def test_clear_binarized_page_reverts_to_source(client, data_root, monkeypatch):
    stem = "folio"
    (data_root / "manuscripts" / "pages" / f"{stem}.jpg").write_bytes(minimal_jpeg_bytes())
    monkeypatch.setattr(
        "annote.services.kraken_binarize.binarize_image",
        lambda image: Image.new("RGB", image.size, "black"),
    )
    client.post(f"/pages/{stem}/binarize")

    response = client.delete(f"/pages/{stem}/binarize")
    assert response.status_code == 200
    assert response.json()["binarized_at"] is None
    assert not processed_page_path(data_root, stem).is_file()
    assert resolve_working_page_image(data_root, stem).name == f"{stem}.jpg"


def test_get_page_image_serves_binarized_when_active(client, data_root, monkeypatch):
    stem = "folio"
    source = data_root / "manuscripts" / "pages" / f"{stem}.jpg"
    source.write_bytes(minimal_jpeg_bytes())

    monkeypatch.setattr(
        "annote.services.kraken_binarize.binarize_image",
        lambda image: Image.new("RGB", image.size, (0, 0, 0)),
    )
    client.post(f"/pages/{stem}/binarize")

    response = client.get(f"/pages/{stem}/image")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == processed_page_path(data_root, stem).read_bytes()


def test_binarize_requires_kraken_message(client, data_root, monkeypatch):
    stem = "folio"
    (data_root / "manuscripts" / "pages" / f"{stem}.jpg").write_bytes(minimal_jpeg_bytes())

    def boom(*_args, **_kwargs):
        raise RuntimeError("Kraken is required for binarization. Install with: pip install 'annote[kraken]'")

    monkeypatch.setattr("annote.api.pages.binarize_page", boom)

    response = client.post(f"/pages/{stem}/binarize")
    assert response.status_code == 400
    assert "Kraken is required" in response.json()["detail"]


def test_auto_segment_preserves_binarized_at(client, data_root, monkeypatch):
    stem = "folio"
    (data_root / "manuscripts" / "pages" / f"{stem}.jpg").write_bytes(minimal_jpeg_bytes())

    monkeypatch.setattr(
        "annote.services.kraken_binarize.binarize_image",
        lambda image: Image.new("RGB", image.size, "black"),
    )
    binarize_response = client.post(f"/pages/{stem}/binarize")
    assert binarize_response.status_code == 200
    binarized_at = binarize_response.json()["binarized_at"]
    assert binarized_at is not None

    used_paths: list[Path] = []

    def fake_segment_image(image, *, device="cpu"):
        used_paths.append(resolve_working_page_image(data_root, stem))
        return refine_kraken_segments(
            image,
            [
                Segment(
                    id="seg-new",
                    number=1,
                    kind="polygon",
                    points=[[1, 1], [9, 1], [9, 4], [1, 4]],
                )
            ],
        )

    monkeypatch.setattr("annote.services.kraken_segment.segment_image", fake_segment_image)

    response = client.post(f"/pages/{stem}/segment", json={"replace": True, "pair_transcription": False})
    assert response.status_code == 200
    body = response.json()
    assert body["binarized_at"] == binarized_at
    assert used_paths == [processed_page_path(data_root, stem)]


def test_binarize_page_service_round_trip(data_root, monkeypatch):
    stem = "folio"
    (data_root / "manuscripts" / "pages" / f"{stem}.jpg").write_bytes(minimal_jpeg_bytes())

    monkeypatch.setattr(
        "annote.services.kraken_binarize.binarize_image",
        lambda image: Image.new("RGB", image.size, (10, 20, 30)),
    )

    saved = binarize_page(data_root, stem)
    assert saved.binarized_at is not None
    cleared = clear_binarized_page(data_root, stem)
    assert cleared.binarized_at is None
