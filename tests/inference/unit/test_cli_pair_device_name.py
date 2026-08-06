"""Whatever names this machine, the platform gets a name it will accept.

`DEVICE_NAME_LIMIT` mirrors the cap `PairingStartRequest` applies on the way in.
It existed so a long hostname could not turn into a 422 a researcher cannot act
on - but it was applied inside `this_machine_name()`, which `--name` bypasses
entirely. A researcher who passed a name over the cap got exactly the
unactionable 422 the limit was written to prevent, and the flag that caused it
was the one thing they had chosen deliberately.

The truncation therefore belongs where the name is settled, not at one of the two
places it can come from.
"""

from __future__ import annotations

import argparse

import pytest

from inference.cli import pair as pair_module
from inference.cli.api import DEVICE_NAME_LIMIT, PlatformError


class _RecordingClient:
    """Accepts a pairing start, records the name, then stops the flow."""

    base_url = "https://api.nomicous.com"

    def __init__(self) -> None:
        self.device_names: list[str] = []

    def start_pairing(self, *, device_name, device_platform, agent_version, capabilities):
        self.device_names.append(device_name)
        raise PlatformError("stop here - the name is all this test needs")


def _pair_with(name: str | None) -> str:
    client = _RecordingClient()
    args = argparse.Namespace(name=name, no_browser=True, api_url=None, force=False)
    with pytest.raises(PlatformError):
        pair_module._pair(None, None, client, args)
    return client.device_names[0]


def test_an_over_long_name_is_truncated_rather_than_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pair_module, "this_machine_platform", lambda: "darwin-arm64")
    monkeypatch.setattr(pair_module, "installed_version", lambda: "0.4.0")

    sent = _pair_with("n" * (DEVICE_NAME_LIMIT + 200))

    assert len(sent) == DEVICE_NAME_LIMIT


def test_a_name_within_the_limit_is_sent_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cap is a backstop, not a rewrite of what the researcher asked for."""
    monkeypatch.setattr(pair_module, "this_machine_platform", lambda: "darwin-arm64")
    monkeypatch.setattr(pair_module, "installed_version", lambda: "0.4.0")

    assert _pair_with("  Ada's laptop  ") == "Ada's laptop"


def test_a_blank_name_still_falls_back_to_the_hostname(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pair_module, "this_machine_platform", lambda: "darwin-arm64")
    monkeypatch.setattr(pair_module, "installed_version", lambda: "0.4.0")
    monkeypatch.setattr(pair_module, "this_machine_name", lambda: "ada-laptop")

    assert _pair_with("   ") == "ada-laptop"


# `test_an_over_long_hostname_is_still_truncated` stood here. `pair.py` truncates at the
# settling point, so `api.py`'s `name[:DEVICE_NAME_LIMIT]` is a second slice of an
# already-sliced string; the test pinned that vestigial guard rather than the behaviour
# this file's docstring is about, which
# `test_an_over_long_name_is_truncated_rather_than_sent` covers through the real flow.
