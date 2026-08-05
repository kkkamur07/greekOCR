"""**Capacity**: whether an **inference host** can take work right now.

Answered from recent device activity, and from nothing else. There is no probe,
no health endpoint, and no "is cloud enabled" flag - a host has capacity when one
of its devices was seen recently, and that is the same sentence for a
researcher's laptop and for a hosted worker (ADR 0003).

The window is ``DEVICE_IDLE_WINDOW_SECONDS``, reused rather than duplicated: it
is already the boundary past which the device layer calls a device ``offline``,
and inventing a second freshness dial would let the two disagree about the same
machine. An ``idle`` device still counts - it is running, it is polling, it will
pick the job up on its next cycle.

**Cloud capacity will not exist for some time.** That needs no special handling:
no ``cloud`` device row means submission finds no capacity there and says so.
When hosted workers are stood up they simply begin answering the same query.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.settings.device import DeviceSettings, get_device_settings
from backend.ml.domain.execution import ExecutionRequest, ExecutionTarget
from backend.ml.infrastructure.device_repository import HelperDeviceRepository
from backend.users.infrastructure.orm_models import User


class InferenceCapacityService:
    def __init__(
        self,
        repository: HelperDeviceRepository | None = None,
        settings: DeviceSettings | None = None,
    ) -> None:
        self._repo = repository or HelperDeviceRepository()
        self._settings = settings

    @property
    def settings(self) -> DeviceSettings:
        # Resolved per call rather than at construction: the routers build this
        # service at import time, and the settings cache is reset between tests.
        return self._settings or get_device_settings()

    async def available_targets(
        self, session: AsyncSession, *, user_id: UUID, now: datetime | None = None
    ) -> frozenset[ExecutionTarget]:
        now = now or datetime.now(UTC)
        seen_after = now - timedelta(seconds=self.settings.device_idle_window_seconds)
        return await self._repo.hosts_with_recent_devices(
            session, user_id=user_id, seen_after=seen_after
        )

    async def execution_request(
        self, session: AsyncSession, user: User, *, now: datetime | None = None
    ) -> ExecutionRequest:
        """Read capacity once, at the top of submission, and freeze it.

        Everything downstream is a pure function of this value. Re-reading
        capacity later would be a second decision, and the whole point of ADR
        0002 is that there is exactly one.
        """
        return ExecutionRequest.for_preference(
            prefer_local=bool(user.prefer_local_inference),
            available=await self.available_targets(session, user_id=user.id, now=now),
        )
