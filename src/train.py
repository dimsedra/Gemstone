import os
import json
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def setup_logger(log_path: str) -> logging.Logger:
    """Configure a logger that writes to both stdout and a log file."""
    logger = logging.getLogger("gemstone_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # File handler — always append so resumed runs are contiguous
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console / stdout handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def save_metrics_plot(history: dict, models_dir: str):
    """Regenerate and overwrite the training metrics PNG."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"],   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curves")

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"],   label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curves")

    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, "training_metrics.png"))
    plt.close()


def train_model():
    # ── Paths ───────────────────────────────────────────────────────────────
    BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir   = os.path.join(BASE_DIR, "data")
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    log_path        = os.path.join(models_dir, "training.log")
    checkpoint_path = os.path.join(models_dir, "gemstone_resnet50.pth")
    history_path    = os.path.join(models_dir, "training_history.json")
    indices_path    = os.path.join(models_dir, "class_indices.json")

    logger = setup_logger(log_path)

    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ── Transforms ──────────────────────────────────────────────────────────
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # ── Datasets & Loaders ──────────────────────────────────────────────────
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"),
                                         transform=train_transform)
    valid_dataset = datasets.ImageFolder(os.path.join(data_dir, "valid"),
                                         transform=valid_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                              num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False,
                              num_workers=2, pin_memory=True)

    num_classes = len(train_dataset.classes)
    logger.info(f"Found {num_classes} classes.")

    # Save / overwrite class-index mapping
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
    with open(indices_path, "w") as f:
        json.dump(idx_to_class, f, indent=4)

    # ── Model ────────────────────────────────────────────────────────────────
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

    # ── Resume or start fresh ────────────────────────────────────────────────
    if os.path.exists(checkpoint_path) and os.path.exists(history_path):
        logger.info("Resuming from existing checkpoint …")
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )
        with open(history_path) as f:
            history = json.load(f)
        start_epoch   = len(history["train_loss"])
        best_val_loss = min(history["val_loss"])
        logger.info(f"Loaded history: {start_epoch} epochs done, "
                    f"best val_loss = {best_val_loss:.4f}")
    else:
        logger.info("Starting fresh training …")
        history = {"train_loss": [], "train_acc": [],
                   "val_loss":   [], "val_acc":   []}
        start_epoch   = 0
        best_val_loss = float("inf")

    # ── Hyper-parameters ─────────────────────────────────────────────────────
    MAX_EPOCHS = 100
    PATIENCE   = 5
    patience_counter = 0

    logger.info(f"Training from epoch {start_epoch + 1} to {MAX_EPOCHS} "
                f"(early stopping patience = {PATIENCE})")
    logger.info("-" * 72)

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, MAX_EPOCHS):
        t0 = time.time()

        # — Train —
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            run_loss += loss.item() * images.size(0)
            _, pred   = outputs.max(1)
            total    += labels.size(0)
            correct  += pred.eq(labels).sum().item()

        train_loss = run_loss / total
        train_acc  = correct  / total

        # — Validate —
        model.eval()
        run_vloss, vcorrect, vtotal = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs   = model(images)
                loss      = criterion(outputs, labels)
                run_vloss += loss.item() * images.size(0)
                _, pred   = outputs.max(1)
                vtotal    += labels.size(0)
                vcorrect  += pred.eq(labels).sum().item()

        val_loss = run_vloss / vtotal
        val_acc  = vcorrect  / vtotal

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch + 1:>3}/{MAX_EPOCHS}  ({elapsed:.1f}s)  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  |  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        # — Checkpoint on improvement —
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"  --> New best val_loss={best_val_loss:.4f}  checkpoint saved!")
        else:
            patience_counter += 1
            logger.info(f"  --> No improvement. Patience {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                logger.info("Early stopping triggered!")
                break

        # — Persist history and redraw chart after every epoch —
        with open(history_path, "w") as f:
            json.dump(history, f, indent=4)
        save_metrics_plot(history, models_dir)

    logger.info("-" * 72)
    logger.info(f"Training complete. Best val_loss = {best_val_loss:.4f}")
    logger.info(f"Log file : {log_path}")
    logger.info(f"Chart    : {os.path.join(models_dir, 'training_metrics.png')}")


if __name__ == "__main__":
    train_model()
