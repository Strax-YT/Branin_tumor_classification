import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "app"))

FIXTURE_FILES = {
    "glioma": "glioma.png",
    "meningioma": "meningioma.png",
    "pituitary": "pituitary.png",
    "no_tumor": "no_tumor.jpg",
}


@pytest.fixture
def sample_image_bytes():
    """Returns a function: sample_image_bytes(class_name) -> raw bytes of a real MRI fixture."""
    def _load(class_name="glioma"):
        path = FIXTURES_DIR / FIXTURE_FILES[class_name]
        return path.read_bytes()
    return _load


@pytest.fixture
def corrupted_image_bytes():
    return b"this is not a real image file, just garbage bytes \x00\x01\x02"


@pytest.fixture
def api_client(monkeypatch, tmp_path):
    """A FastAPI TestClient wired to a throwaway SQLite DB, isolated from neurosight.db."""
    from api import db as db_module

    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test_neurosight.db")

    from fastapi.testclient import TestClient
    from api.main import app

    with TestClient(app) as client:
        yield client
