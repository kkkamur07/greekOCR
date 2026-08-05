"""One registry of every memoized settings accessor.

Each settings class is built once per process behind an ``lru_cache``d accessor.
Anything that changes the environment after those objects exist - a test that
flips ``ENVIRONMENT`` to production, a fixture that repoints ``STORAGE_BACKEND``
at Supabase - has to invalidate *every* one of them, because a class it forgets
keeps serving values read from the old environment.

Enumerating the accessors by hand is the failure this exists to remove. Three
call sites do it today and two of them are incomplete, so the classes they miss
are precisely the ones nobody clears. Enrollment happens at decoration time
instead, which makes the list impossible to get wrong: a settings module that
has been imported is registered, and one that has not been imported has no cache
to clear.

This does not change how any individual accessor behaves. ``settings_cache`` is
``functools.lru_cache`` with a side effect, so ``get_app_settings.cache_clear()``
and friends keep working exactly as before.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import _lru_cache_wrapper, lru_cache
from typing import TypeVar

_SettingsT = TypeVar("_SettingsT")

#: Populated by decoration, so it tracks imports rather than a hand-kept list.
_REGISTERED: list[_lru_cache_wrapper] = []


def settings_cache(build: Callable[[], _SettingsT]) -> _lru_cache_wrapper[_SettingsT]:
    """Memoize a settings accessor and enroll it in :func:`reset_settings_caches`."""
    accessor = lru_cache(build)
    _REGISTERED.append(accessor)
    return accessor


def reset_settings_caches() -> None:
    """Drop every memoized settings object so the next read re-parses the environment.

    Use this instead of clearing accessors one at a time. Validation still runs on
    the next access, not here, so a reset never raises on bad configuration - the
    caller sees the failure at the point it asks for settings, exactly as a fresh
    process would.
    """
    for accessor in _REGISTERED:
        accessor.cache_clear()
