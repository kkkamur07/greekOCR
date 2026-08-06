"""The shipped development template has to produce a runnable configuration.

Four documents tell a new contributor to copy ``backend/core/.env.example`` to
``backend/core/.env`` and start the API. Nothing checked that the result loads,
and it did not: ``JOB_WORKER_CLAIM_TEST_ONLY=`` reached pydantic as ``''``, which
is not a ``bool | None``, and ``create_app()`` resolves the job settings before
it registers a single route. Every reader who followed the instructions got a
``ValidationError`` at boot.

So this loads the file the documents name, through the settings classes the
application actually uses.
"""

from __future__ import annotations

import pytest

from backend.core.settings.app import AppSettings
from backend.core.settings.auth import AuthSettings
from backend.core.settings.device import DeviceSettings
from backend.core.settings.infrastructure import InfrastructureSettings
from backend.core.settings.job import JobSettings
from backend.core.settings.ml import MLSettings
from backend.core.settings.storage import StorageSettings
from tests.fixtures.paths import REPO_ROOT

ENV_EXAMPLE = REPO_ROOT / "nomicous" / "backend" / "core" / ".env.example"

#: Every settings class `create_app` resolves before it will serve a request.
SETTINGS_CLASSES = [
    AppSettings,
    AuthSettings,
    DeviceSettings,
    InfrastructureSettings,
    JobSettings,
    MLSettings,
    StorageSettings,
]


@pytest.fixture
def isolated_env(monkeypatch) -> None:
    """Read the template and nothing else.

    Ambient environment variables win over a .env file, so a value the test
    session happens to export would hide a broken line in the template.
    """
    for settings_class in SETTINGS_CLASSES:
        for field in settings_class.model_fields.values():
            if field.alias:
                monkeypatch.delenv(field.alias, raising=False)


@pytest.mark.parametrize("settings_class", SETTINGS_CLASSES, ids=lambda cls: cls.__name__)
def test_every_settings_class_loads_the_shipped_template(isolated_env, settings_class) -> None:
    settings_class(_env_file=ENV_EXAMPLE)
