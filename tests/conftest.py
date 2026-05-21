"""Shared pytest configuration."""

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: requires Postgres (docker compose up db)")
