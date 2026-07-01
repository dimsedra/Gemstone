import os
import json
import pytest
from fastapi.testclient import TestClient

from src.app import app

client = TestClient(app)

def test_get_info():
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "trained" in data
    if data["trained"]:
        assert "total_classes" in data
        assert "classes" in data
        assert "metrics_chart_available" in data
    else:
        assert data["trained"] is False
        assert "classes" in data
        assert len(data["classes"]) == 0

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
    from src.app import static_dir, models_dir

    # Write a temporary file to static directory
    os.makedirs(static_dir, exist_ok=True)
    temp_static_file = os.path.join(static_dir, "test_static_mount.txt")
    with open(temp_static_file, "w") as f:
        f.write("static content")
    
    # Write a dummy training_metrics.png file in the absolute models directory
    os.makedirs(models_dir, exist_ok=True)
    dummy_metrics_file = os.path.join(models_dir, "training_metrics.png")
    
    # Backup real training_metrics.png if it exists
    metrics_backup = None
    if os.path.exists(dummy_metrics_file):
        with open(dummy_metrics_file, "rb") as f:
            metrics_backup = f.read()

    with open(dummy_metrics_file, "w") as f:
        f.write("dummy png content")

    # Write a temporary other file in the models directory to test 404
    temp_weights_file = os.path.join(models_dir, "some_weights.pth")
    with open(temp_weights_file, "w") as f:
        f.write("weights content")
        
    try:
        response_static = client.get("/static/test_static_mount.txt")
        assert response_static.status_code == 200
        assert response_static.text == "static content"
        
        # Test metrics chart endpoint
        response_metrics = client.get("/models/training_metrics.png")
        assert response_metrics.status_code == 200
        assert response_metrics.text == "dummy png content"

        # Test arbitrary file in models directory returns 404
        response_weights = client.get("/models/some_weights.pth")
        assert response_weights.status_code == 404
    finally:
        # Clean up
        if os.path.exists(temp_static_file):
            os.remove(temp_static_file)
        if os.path.exists(temp_weights_file):
            os.remove(temp_weights_file)
        
        # Restore or clean up training_metrics.png
        if metrics_backup is not None:
            with open(dummy_metrics_file, "wb") as f:
                f.write(metrics_backup)
        else:
            if os.path.exists(dummy_metrics_file):
                os.remove(dummy_metrics_file)

class MockClassifier:
    def __init__(self, model_path=None, mapping_path=None):
        pass
    def predict(self, file_like):
        return {
            "prediction": "Ruby",
            "confidence": 0.95,
            "top_5": [
                {"class": "Ruby", "confidence": 0.95},
                {"class": "Sapphire", "confidence": 0.03},
                {"class": "Emerald", "confidence": 0.01},
                {"class": "Diamond", "confidence": 0.005},
                {"class": "Quartz", "confidence": 0.005}
            ]
        }

def test_predict_model_not_available(monkeypatch):
    import src.app
    monkeypatch.setattr(src.app, "classifier", None)
    
    # Mock os.path.exists to return False for the model path
    import os
    original_exists = os.path.exists
    def mock_exists(path):
        if "gemstone_resnet50.pth" in path:
            return False
        return original_exists(path)
    monkeypatch.setattr(os.path, "exists", mock_exists)
    
    response = client.post("/predict", files={"file": ("test.jpg", b"dummy image data", "image/jpeg")})
    assert response.status_code == 503
    assert response.json()["detail"] == "Model is not trained/available."

def test_predict_not_an_image():
    response = client.post("/predict", files={"file": ("test.txt", b"dummy text data", "text/plain")})
    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is not an image."

def test_predict_none_mime_type(monkeypatch):
    import src.app
    monkeypatch.setattr(src.app, "classifier", MockClassifier())
    response = client.post("/predict", files={"file": ("test", b"fake_image_bytes", None)})
    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is not an image."



def test_predict_success(monkeypatch):
    import src.app
    monkeypatch.setattr(src.app, "classifier", MockClassifier())
    
    response = client.post("/predict", files={"file": ("test.jpg", b"fake_image_bytes", "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["prediction"] == "Ruby"
    assert data["confidence"] == 0.95
    assert len(data["top_5"]) == 5


