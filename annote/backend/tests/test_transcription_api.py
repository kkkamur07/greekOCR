"""Page transcription API."""

from annote.services.text_lines import split_text_lines


def test_get_transcription_returns_parsed_lines(client, data_root):
    text = "αἱρετικῶν\nκαὶ φιλοσόφων\n"
    (data_root / "transcriptions" / "pages" / "folio.txt").write_text(text, encoding="utf-8")
    (data_root / "manuscripts" / "pages" / "folio.jpg").write_bytes(b"\xff\xd8\xff")

    response = client.get("/pages/folio/transcription")

    assert response.status_code == 200
    body = response.json()
    assert body["raw_text"] == text
    assert body["text_lines"] == [
        {"index": 1, "text": "αἱρετικῶν"},
        {"index": 2, "text": "καὶ φιλοσόφων"},
    ]


def test_get_transcription_missing_file(client, data_root):
    (data_root / "manuscripts" / "pages" / "folio.jpg").write_bytes(b"\xff\xd8\xff")

    response = client.get("/pages/folio/transcription")

    assert response.status_code == 200
    body = response.json()
    assert body["raw_text"] is None
    assert body["text_lines"] == []
    assert body["status"] == "missing"
