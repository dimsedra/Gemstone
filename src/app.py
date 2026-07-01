import os
import json
import threading
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from src.inference import GemstoneClassifier

app = FastAPI(title="Gemstone Classification System")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
static_dir = os.path.join(BASE_DIR, "static")
models_dir = os.path.join(BASE_DIR, "models")

# Ensure directories exist prior to Starlette StaticFiles instantiation
os.makedirs(static_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Initialize classifier lazily on startup
classifier = None
classifier_lock = threading.Lock()

@app.on_event("startup")
def load_classifier():
    global classifier
    os.makedirs(static_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "gemstone_resnet50.pth")
    mapping_path = os.path.join(models_dir, "class_indices.json")
    
    if not os.path.exists(model_path) or not os.path.exists(mapping_path):
        print(f"Warning: Model or class indices not found. Make sure to train the model first.")
    else:
        classifier = GemstoneClassifier(model_path, mapping_path)
        print("Classifier loaded successfully.")

@app.get("/")
def get_index():
    index_path = os.path.join(static_dir, "index.html")
    if not os.path.exists(index_path):
        return {"message": "Frontend index.html not found. Place it in /static folder."}
    return FileResponse(index_path)

# Model Prediction endpoint
@app.post("/predict")
def predict_gemstone(file: UploadFile = File(...)):
    global classifier
    if classifier is None:
        with classifier_lock:
            if classifier is None:
                # Try to initialize if it wasn't initialized
                model_path = os.path.join(models_dir, "gemstone_resnet50.pth")
                mapping_path = os.path.join(models_dir, "class_indices.json")
                if os.path.exists(model_path) and os.path.exists(mapping_path):
                    os.makedirs(static_dir, exist_ok=True)
                    os.makedirs(models_dir, exist_ok=True)
                    classifier = GemstoneClassifier(model_path, mapping_path)
                else:
                    raise HTTPException(status_code=503, detail="Model is not trained/available.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    try:
        # Run prediction directly from uploaded file stream
        result = classifier.predict(file.file)
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

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

# Serve rest of static files (css, js, images)
app.mount("/static", StaticFiles(directory=static_dir), name="static")
# Serve models directory as static so we can load the metrics chart in frontend
app.mount("/models", StaticFiles(directory=models_dir), name="models")
