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

def test_static_mounts():
    # Write a temporary file to static directory
    os.makedirs("static", exist_ok=True)
    temp_static_file = os.path.join("static", "test_static_mount.txt")
    with open(temp_static_file, "w") as f:
        f.write("static content")
    
    # Write a temporary file to models directory
    os.makedirs("models", exist_ok=True)
    temp_models_file = os.path.join("models", "test_models_mount.txt")
    with open(temp_models_file, "w") as f:
        f.write("models content")
        
    try:
        response_static = client.get("/static/test_static_mount.txt")
        assert response_static.status_code == 200
        assert response_static.text == "static content"
        
        response_models = client.get("/models/test_models_mount.txt")
        assert response_models.status_code == 200
        assert response_models.text == "models content"
    finally:
        # Clean up
        if os.path.exists(temp_static_file):
            os.remove(temp_static_file)
        if os.path.exists(temp_models_file):
            os.remove(temp_models_file)


