"""annote — standalone manuscript line annotation API."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("annote")
except PackageNotFoundError:
    __version__ = "0.0.0"
