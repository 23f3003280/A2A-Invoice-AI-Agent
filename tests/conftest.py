import os
import tempfile
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from fastapi.testclient import TestClient

@pytest.fixture()
def client(monkeypatch, tmp_path):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    # Reload app modules so the engine uses the test database.
    import importlib
    import app.config, app.database, app.models, app.main
    importlib.reload(app.config)
    importlib.reload(app.database)
    importlib.reload(app.models)
    importlib.reload(app.main)
    app.database.Base.metadata.create_all(bind=app.database.engine)
    with TestClient(app.main.app) as c:
        yield c
