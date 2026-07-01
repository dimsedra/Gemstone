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
