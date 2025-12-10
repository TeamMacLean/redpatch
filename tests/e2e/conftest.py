"""
Pytest configuration for Playwright E2E tests.
"""

import pytest


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "e2e: mark test as end-to-end test")


# Re-export playwright fixtures
pytest_plugins = ["playwright.pytest_plugin"]
