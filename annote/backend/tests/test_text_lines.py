"""Text line parser — split page transcription on line breaks."""

from annote.services.text_lines import split_text_lines


def test_split_text_lines_on_line_breaks():
    text = "αἱρετικῶν\nκαὶ φιλοσόφων\n"
    lines = split_text_lines(text)
    assert [line.text for line in lines] == ["αἱρετικῶν", "καὶ φιλοσόφων"]
    assert [line.index for line in lines] == [1, 2]


def test_split_text_lines_strips_trailing_empty_line():
    lines = split_text_lines("line one\nline two\n\n")
    assert [line.text for line in lines] == ["line one", "line two"]


def test_split_text_lines_empty_file():
    assert split_text_lines("") == []
    assert split_text_lines("   \n  ") == []
