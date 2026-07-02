# Gemstone Image Classification System

An AI-powered local gemstone identification dashboard using PyTorch (ResNet50) for transfer learning and a FastAPI backend with a sleek glassmorphic dark-mode frontend.

For the full academic research findings, per-class metrics, and analysis, please refer to the [Research Documentation](docs/research/research_documentation.md).

---

## 1. Project Directory Structure

```text
gemstone/
├── .venv/                     # Python Virtual Environment (git-ignored)
├── data/                      # Dataset folders (train, valid, test)
├── docs/                      # Project documentation
│   └── research/
│       └── research_documentation.md # Detailed research report & paper analysis
├── models/                    # Trained weights and outputs (git-ignored)
│   ├── gemstone_resnet50.pth  # Best model checkpoint
│   ├── class_indices.json     # 87 class index to gemstone name mapping
│   └── training_metrics.png   # Train vs. Val loss and accuracy curves
├── src/
│   ├── train.py               # Fine-tuning ResNet50 script
│   ├── evaluate.py            # Model test set evaluation script
│   ├── inference.py           # Core classification helper class
│   └── app.py                 # FastAPI backend server
├── static/                    # Frontend files
│   ├── index.html             # Sleek dark-mode dashboard
│   ├── style.css              # Custom styling (glassmorphism details)
│   └── script.js              # Interactivity (upload, drag-drop, AJAX API)
├── tests/
│   └── test_app.py            # Backend API test suite
├── requirements.txt           # Package dependencies
└── walkthrough.md             # Detailed implementation and metrics walkthrough
```

---

## 2. Installation and Setup

To run this application locally, follow these setup steps:

### Step A: Clone the Repository & Create Virtual Environment
1. **Open your terminal or PowerShell** in the project's root directory:
   ```powershell
   # Create a virtual environment
   python -m venv .venv

   # Activate the virtual environment
   # On PowerShell:
   .venv\Scripts\Activate.ps1
   # On CMD:
   .venv\Scripts\activate.bat
   ```
2. **Install all required dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

### Step B: Download Dataset & Model Weights
1. Open the [Google Drive Folder](https://drive.google.com/drive/folders/1R_nfCIM8dykIqpB1ORn9C_ljeLn__4u9?usp=sharing).
2. **Download the model weights** (`gemstone_resnet50.pth` and `training_history.json`) and place them inside the `models/` folder.
3. **Download the dataset folders** (`train`, `valid`, `test`) and place them inside the `data/` folder so the path structure matches `data/train/`, `data/valid/`, and `data/test/`.

---

## 3. How to Run the Web Application

Once the setup steps are complete:

1. **Start the Uvicorn web server** (ensure your virtual environment is active):
   ```powershell
   uvicorn src.app:app --reload
   ```

2. **Open your web browser** and navigate to:
   ```text
   http://127.0.0.1:8000/
   ```

3. **Upload or drag-and-drop** gemstone images to classify them instantly!

---

## 4. How to Train or Evaluate the Model (Optional)

If you modify the dataset or want to retrain/test the model, you can run the following scripts:

### Step A: Activate the Virtual Environment
```powershell
# PowerShell
.venv\Scripts\Activate.ps1

# CMD
.venv\Scripts\activate.bat
```

### Step B: Train the Model
To start training the ResNet50 model using transfer learning:
```powershell
python src/train.py
```
*This will run for up to 100 epochs with early stopping (patience = 5). It will automatically save the best model weights to `models/gemstone_resnet50.pth` and curves to `models/training_metrics.png`.*

#### Hardware Requirements & VRAM Benchmarks (RTX 3060 12GB)
Training is optimized for NVIDIA CUDA-enabled GPUs, but will fall back to CPU if unavailable. Below is the VRAM usage profile based on the **GeForce RTX 3060 (12GB VRAM)**:

| Mode | Batch Size | VRAM Usage | Notes |
|---|---|---|---|
| **Frozen Backbone** (Current) | 64 | **~2.8 GB** | Very lightweight. Fits easily on 4GB+ GPUs. |
| **Fully Unfrozen** (Fine-tuning) | 64 | **~3.5 GB** | Recommended only if fine-tuning backbone layers. |
| **Frozen Backbone** (Large Batch) | 256 | **~8.2 GB** | Faster training. Best for 8GB+ VRAM GPUs. |

**Recommendations:**
- **Local GPU Training**: A local GPU with **at least 4 GB VRAM** (e.g., GTX 1660, RTX 3050) is highly recommended.
- **Cloud Training**: If your local machine lacks a dedicated GPU or has less than 4 GB VRAM, it is advised to run the training script in a cloud environment (e.g., Google Colab, Kaggle Notebooks, or Lambda Labs) utilizing a free T4 GPU.
- **Batch Size Scaling**: If you experience out-of-memory (OOM) errors on smaller GPUs, open `src/train.py` and lower `batch_size` in the DataLoader from `64` to `32` or `16`.

### Step C: Evaluate on the Test Set
To calculate model classification metrics and overall accuracy on the test set:
```powershell
python src/evaluate.py
```

### Step D: Run Backend Tests
To run the automated test suite using Pytest:
```powershell
pytest
```
