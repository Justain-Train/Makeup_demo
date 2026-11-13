import pytest

from app.config import get_settings


@pytest.fixture(autouse=True)
def _rate_limit_off(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
