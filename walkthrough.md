# Project Walkthrough: Gemstone Image Classification

We have completed the implementation of the local gemstone classification system using PyTorch transfer learning and FastAPI. The code has been fully tested and successfully pushed to your GitHub repository: [https://github.com/dimsedra/Gemstone](https://github.com/dimsedra/Gemstone).

---

## 1. What Was Accomplished

We created a structured local computer vision project with the following components:

### A. Environment & Setup
- Configured a Python virtual environment `.venv` with PyTorch, Torchvision, FastAPI, Pillow, Matplotlib, and Scikit-Learn.
- Installed CUDA-enabled PyTorch (`2.5.1+cu121`) ensuring the training runs on your local **NVIDIA GeForce RTX 3060** GPU.

### B. PyTorch Training Pipeline (`src/train.py`)
- Fine-tuned **ResNet50** via transfer learning (frozen backbone parameters, trained classification head).
- Applied random flips and 15-degree rotations on training images to avoid overfitting.
- Dynamically resolved class mappings for all 87 gemstone categories.
- Saved best validation checkpoint to `models/gemstone_resnet50.pth`, mapping index to name in `models/class_indices.json`, and training curves to `models/training_metrics.png`.
- Automated headless `'Agg'` backend support for matplotlib plotting.

### C. Evaluation Suite (`src/evaluate.py`)
- Evaluated the trained weights on the `data/test` directory.
- Remapped test labels back to training index coordinates to prevent alphabetically-shifted mapping issues.
- Overall test accuracy reached **65.60%** with flawless classification performance (100.0% F1-score) on categories like `Ametrine`, `Blue Lace Agate`, and `Chrysoprase`.

### D. Reusable Inference Helper (`src/inference.py`)
- Implemented `GemstoneClassifier` with secure `weights_only=True` PyTorch loading.
- Applied `ImageOps.exif_transpose` to handle phone camera image rotations dynamically.
- Returns a JSON-ready structure with top predicted class and confidence alongside Top-5 alternative matches.

### E. FastAPI Backend API (`src/app.py`)
- Created a robust web server running locally with `/predict`, `/info`, and static file serving.
- Implemented **thread-safe lazy model loading** using `threading.Lock()` to prevent concurrent startup race conditions.
- Used a synchronous route signature for the predict endpoint so FastAPI automatically offloads blocking model execution to an external threadpool.
- Secured model files by removing public mounts, exposing only the metrics chart through a dedicated route.

### F. Sleek Glassmorphic Web Dashboard (`static/`)
- **`index.html`**: A responsive dashboard containing drag-and-drop file upload, image previews, dynamic result cards, a supported gemstone search list, and metrics tab.
- **`style.css`**: Vanilla CSS variable configuration styling cards with glassmorphism blurred backdrops, customized scrollbars, and colored confidence bars.
- **`script.js`**: Handles upload forms, tab transitions, search filters, and AJAX requests.

---

## 2. Test Verification Summary

We built a test suite with 8 unit and integration tests inside `tests/test_app.py` covering:
- `/info` metadata retrieval (both trained and untrained states).
- HTML serving fallbacks.
- Static file mount accessibility.
- Upload validation (rejecting non-image files, handling missing MIME types).
- Inference mock pipeline verification.

All tests pass cleanly:
```text
======================== 8 passed, 3 warnings in 5.47s ========================
```

---

## 3. Training Progress Visualization

The training metrics curves are saved in `models/training_metrics.png` (and served via the dashboard), showing:
- **Loss**: Training loss dropped to **0.2414**, while validation loss settled at **1.0274**.
- **Accuracy**: Training accuracy reached **92.84%**, with validation accuracy reaching **69.50%**.
