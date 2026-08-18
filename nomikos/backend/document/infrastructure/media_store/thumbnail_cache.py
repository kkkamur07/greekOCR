"""Bounded in-process cache for rendered part thumbnails.

Thumbnails are pure functions of (source bytes, target width), and rendering one costs a
full decode of a manuscript scan plus a LANCZOS resize. The public media route serves
them without authentication, so without a cache every repeated request pays that cost
again - a few bytes of query string amplified into hundreds of megapixels of work.

Keyed on a digest of the source bytes rather than on the storage key, so the cache stays
correct when a part's image is replaced under the same key, and so an entry is never
tied to a document whose visibility may change (authorization happens before the read;
this layer only memoises pixels).

The cache is per process. Nothing here is shared between workers on purpose: it protects
CPU, which is a per-process resource, and a shared store would trade the decode for a
network round trip on every hit.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from hashlib import blake2b

#: Total rendered bytes retained. Thumbnails are tens of kilobytes, so this holds a few
#: hundred distinct variants - far more than a published document's page list needs.
MAX_THUMBNAIL_CACHE_BYTES = 32 * 1024 * 1024

_lock = threading.Lock()
_entries: OrderedDict[tuple[bytes, int], bytes] = OrderedDict()
_total_bytes = 0


def thumbnail_cache_key(data: bytes, width: int) -> tuple[bytes, int]:
    # blake2b is used as a content fingerprint, not a security primitive; 16 bytes is
    # ample to make an accidental collision between two page images impossible in
    # practice, and it hashes an 8 MB scan far faster than SHA-256.
    return blake2b(data, digest_size=16).digest(), width


def get_cached_thumbnail(key: tuple[bytes, int]) -> bytes | None:
    with _lock:
        cached = _entries.get(key)
        if cached is None:
            return None
        _entries.move_to_end(key)
        return cached


def store_cached_thumbnail(key: tuple[bytes, int], encoded: bytes) -> None:
    """Insert an entry, evicting least-recently-used ones to stay inside the budget."""
    global _total_bytes
    size = len(encoded)
    if size > MAX_THUMBNAIL_CACHE_BYTES:
        # A single entry larger than the whole budget would evict everything else on
        # every insert; skip it rather than let one outlier thrash the cache.
        return
    with _lock:
        previous = _entries.pop(key, None)
        if previous is not None:
            _total_bytes -= len(previous)
        _entries[key] = encoded
        _total_bytes += size
        while _total_bytes > MAX_THUMBNAIL_CACHE_BYTES:
            _, evicted = _entries.popitem(last=False)
            _total_bytes -= len(evicted)


def clear_thumbnail_cache() -> None:
    """Drop every entry - used by tests that assert on render counts."""
    global _total_bytes
    with _lock:
        _entries.clear()
        _total_bytes = 0
