"""What may be written to `device.json`, and what must survive writing it.

The file holds a 180-day **device token** and the platform it is good against,
and `nomicous run` reads both back on every launch. Two properties follow.

**The platform it names must be one this CLI would talk to.** `PlatformClient`
refuses a cleartext remote URL, but refusing it there only would let a single
`nomicous pair --api-url http://...` write a record that every later run reads
back and trusts - laundering a URL that was rejected once into one that is never
questioned again.

**The write is atomic or it did not happen.** The module's docstring promises an
interrupted write leaves the previous credential intact. Through a temporary file
and `os.replace` that held for a killed process, but not for a lost power supply:
the rename could reach the disk before the bytes did, leaving a zero-length
`device.json` that `load_credential` reports as a corrupt credential rather than
as an unpaired machine - a state no `nomicous pair` run recovers from without a
manual delete.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from inference.cli import credentials as credentials_module
from inference.cli.api import InsecurePlatformURL
from inference.cli.credentials import (
    CREDENTIAL_FILE_MODE,
    DeviceCredential,
    file_mode,
    load_credential,
    save_credential,
)


def _credential(platform_url: str = "https://api.nomicous.com") -> DeviceCredential:
    return DeviceCredential(
        platform_url=platform_url,
        device_id="device-1",
        device_token="secret-token",
        account_email="researcher@example.com",
        device_name="laptop",
        token_expires_at=datetime.now(UTC) + timedelta(days=180),
        paired_at=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# Which platforms a token may be stored for
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "platform_url",
    [
        "http://api.nomicous.com",
        "http://staging.internal:8000",
        "http://localhost@evil.example",
        "ftp://api.nomicous.com",
        "",
    ],
)
def test_a_token_is_not_persisted_for_a_platform_it_could_not_be_sent_to(
    platform_url: str, tmp_path: Path
) -> None:
    target = tmp_path / "device.json"

    with pytest.raises(InsecurePlatformURL):
        save_credential(_credential(platform_url), target)

    assert not target.exists()


def test_a_refused_write_leaves_no_temporary_file_behind(tmp_path: Path) -> None:
    """The refusal happens before anything is opened, not halfway through."""
    target = tmp_path / "device.json"

    with pytest.raises(InsecurePlatformURL):
        save_credential(_credential("http://api.nomicous.com"), target)

    assert list(tmp_path.iterdir()) == []


def test_a_refused_write_does_not_disturb_the_credential_already_there(
    tmp_path: Path,
) -> None:
    """A bad `--api-url` must not cost a researcher the pairing they had."""
    target = tmp_path / "device.json"
    save_credential(_credential(), target)

    with pytest.raises(InsecurePlatformURL):
        save_credential(_credential("http://api.nomicous.com"), target)

    stored = load_credential(target)
    assert stored is not None
    assert stored.platform_url == "https://api.nomicous.com"


def test_a_loopback_pairing_still_stores_its_credential(tmp_path: Path) -> None:
    """What the integration suite does, and it must keep working.

    It stands a platform up on `http://127.0.0.1:<port>` and pairs against it.
    Nothing leaves the machine, so the token is not exposed by the scheme.
    """
    target = tmp_path / "device.json"

    save_credential(_credential("http://127.0.0.1:54321"), target)

    stored = load_credential(target)
    assert stored is not None
    assert stored.platform_url == "http://127.0.0.1:54321"


# ---------------------------------------------------------------------------
# Durability
# ---------------------------------------------------------------------------
def test_the_bytes_reach_the_disk_before_the_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ordering the atomic-rename promise actually depends on.

    Without the `fsync`, `os.replace` orders the rename against the file's
    metadata and not against its contents, so a crash between them leaves a
    zero-length `device.json`. This pins the ordering rather than the syscall:
    every fsync of the file descriptor must happen before the rename.
    """
    events: list[str] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def _fsync(fd):
        events.append("fsync")
        return real_fsync(fd)

    def _replace(source, destination):
        events.append("replace")
        return real_replace(source, destination)

    monkeypatch.setattr(credentials_module.os, "fsync", _fsync)
    monkeypatch.setattr(credentials_module.os, "replace", _replace)

    save_credential(_credential(), tmp_path / "device.json")

    assert "fsync" in events, "the credential was renamed into place unflushed"
    assert events.index("fsync") < events.index("replace")


def test_a_directory_that_cannot_be_fsynced_is_not_an_error(tmp_path: Path) -> None:
    """The rename fsync is best effort, deliberately.

    Directory `fsync` is meaningless on Windows and refused outright by some
    network filesystems. Neither is a reason to fail a pairing whose file is
    already written, flushed, and renamed into place.
    """
    credentials_module._fsync_directory(tmp_path / "does-not-exist")


def test_the_rename_is_still_what_publishes_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A write that dies before the rename leaves the old credential in place."""
    target = tmp_path / "device.json"
    save_credential(_credential(), target)

    def _die(source, destination):
        raise OSError("interrupted")

    monkeypatch.setattr(credentials_module.os, "replace", _die)
    with pytest.raises(OSError):
        save_credential(_credential("https://staging.nomicous.com"), target)

    stored = load_credential(target)
    assert stored is not None
    assert stored.platform_url == "https://api.nomicous.com"


def test_the_credential_lands_complete_and_owner_only(tmp_path: Path) -> None:
    """The durability work must not have cost the mode bits or the content."""
    target = tmp_path / "device.json"

    save_credential(_credential(), target)

    stored = load_credential(target)
    assert stored is not None
    assert stored.device_token == "secret-token"
    assert file_mode(target) == CREDENTIAL_FILE_MODE
    assert list(tmp_path.glob(".device.json.tmp")) == []
