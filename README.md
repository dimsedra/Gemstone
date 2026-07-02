# Gemstone Image Classification System

An AI-powered local gemstone identification dashboard using PyTorch (ResNet50) for transfer learning and a FastAPI backend with a sleek glassmorphic dark-mode frontend.

---

## 1. Project Directory Structure

```text
gemstone/
├── .venv/                     # Python Virtual Environment (git-ignored)
├── data/                      # Dataset folders (train, valid, test)
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
To start training the ResNet50 model using transfer learning on your local GPU (GeForce RTX 3060) or CPU:
```powershell
python src/train.py
```
*This will run for up to 100 epochs with early stopping (patience = 5). It will automatically save the best model weights to `models/gemstone_resnet50.pth` and curves to `models/training_metrics.png`.*

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
