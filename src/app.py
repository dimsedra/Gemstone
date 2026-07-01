import os
import json
from fastapi import FastAPI
from fastapi.responses import FileResponse

app = FastAPI(title="Gemstone Classification System")

static_dir = "static"
models_dir = "models"
os.makedirs(static_dir, exist_ok=True)

@app.get("/")
def get_index():
    index_path = os.path.join(static_dir, "index.html")
    if not os.path.exists(index_path):
        return {"message": "Frontend index.html not found. Place it in /static folder."}
    return FileResponse(index_path)

@app.get("/info")
def get_info():
    mapping_path = os.path.join(models_dir, "class_indices.json")
    if not os.path.exists(mapping_path):
        return {"trained": False, "classes": []}
    
    with open(mapping_path, "r") as f:
        idx_to_class = json.load(f)
    
    classes_sorted = [idx_to_class[str(i)] for i in range(len(idx_to_class))]
    metrics_exists = os.path.exists(os.path.join(models_dir, "training_metrics.png"))
    
    return {
        "trained": True,
        "total_classes": len(classes_sorted),
        "classes": classes_sorted,
        "metrics_chart_available": metrics_exists
    }
