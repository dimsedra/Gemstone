import os
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # Paths
    test_dir = "data/test"
    models_dir = "models"
    
    # Load class indices from training
    with open(os.path.join(models_dir, "class_indices.json"), "r") as f:
        idx_to_class = json.load(f)
    
    class_to_idx = {v: int(k) for k, v in idx_to_class.items()}
    classes = [idx_to_class[str(i)] for i in range(len(idx_to_class))]

    # Transforms
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    
    # Remap test targets to align with training class indices
    aligned_samples = []
    for path, target in test_dataset.samples:
        class_name = test_dataset.classes[target]
        if class_name in class_to_idx:
            correct_label = class_to_idx[class_name]
            aligned_samples.append((path, correct_label))
        else:
            print(f"Warning: Class '{class_name}' from test set not found in training class indices.")
    
    test_dataset.samples = aligned_samples
    test_dataset.targets = [s[1] for s in aligned_samples]

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    # Load model structure
    model = resnet50()
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, len(classes))
    )
    model.load_state_dict(torch.load(os.path.join(models_dir, "gemstone_resnet50.pth"), map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\nOverall Test Accuracy: {acc * 100:.2f}%")

    # Generate classification report
    report = classification_report(
        all_labels, 
        all_preds, 
        labels=list(range(len(classes))), 
        target_names=classes, 
        output_dict=True, 
        zero_division=0
    )
    
    # Print top 5 and bottom 5 classes by f1-score
    class_scores = [(name, data['f1-score']) for name, data in report.items() if name in classes]
    class_scores.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 classes by F1-Score:")
    for name, score in class_scores[:5]:
        print(f"  {name}: {score:.4f}")
        
    print("\nBottom 5 classes by F1-Score:")
    for name, score in class_scores[-5:]:
        print(f"  {name}: {score:.4f}")

if __name__ == "__main__":
    evaluate_model()
