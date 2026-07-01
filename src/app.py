import os
import json
from fastapi import FastAPI

app = FastAPI(title="Gemstone Classification System")

models_dir = "models"

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
