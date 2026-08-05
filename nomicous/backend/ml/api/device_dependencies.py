"""Feature gate for the whole device layer.

Applied as a router-level dependency on all three device routers rather than as
a mounting condition in ``create_app()``, for two reasons:

* the routes stay in the OpenAPI schema, so a schema test can prove they are
  mounted without the test having to know the flag's value;
* the flag is read per request, so flipping ``DEVICE_PAIRING_ENABLED`` takes
  effect on the next request rather than on the next code change.

Router-level dependencies are solved before the endpoint's own parameters, so a
disabled device layer answers without ever opening a database session.
"""

from __future__ import annotations

from fastapi import HTTPException, status

from backend.core.settings.device import get_device_settings


def require_device_pairing_enabled() -> None:
    """404 the entire device surface while the feature is switched off.

    404 rather than 503: a disabled feature should be indistinguishable from one
    that was never deployed. It also matches the error envelope, which discards
    ``detail`` and substitutes a fixed public message per status code.
    """
    if not get_device_settings().pairing_enabled():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Device pairing is not enabled",
        )
