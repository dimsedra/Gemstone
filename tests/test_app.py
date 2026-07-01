import os
import json
import pytest
from fastapi.testclient import TestClient

# We will import the app. Since src.app doesn't exist, this will fail.
from src.app import app

client = TestClient(app)

def test_get_info():
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "trained" in data
    assert "total_classes" in data
    assert "classes" in data
    assert "metrics_chart_available" in data

def test_get_root_missing_index(tmp_path, monkeypatch):
    # Temporarily change static_dir in app to a non-existent or empty folder
    import src.app
    # Save the original and patch it
    orig_static_dir = src.app.static_dir
    monkeypatch.setattr(src.app, "static_dir", str(tmp_path))
    
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Frontend index.html not found. Place it in /static folder."}

def test_get_root_with_index(tmp_path, monkeypatch):
    # Temporarily place an index.html in a temp static directory
    import src.app
    monkeypatch.setattr(src.app, "static_dir", str(tmp_path))
    
    index_file = tmp_path / "index.html"
    index_file.write_text("Hello World html")
    
    response = client.get("/")
    assert response.status_code == 200
    assert response.text == "Hello World html"

