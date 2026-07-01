import os
import json
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.models import resnet50

class GemstoneClassifier:
    def __init__(self, model_path="models/gemstone_resnet50.pth", class_mapping_path="models/class_indices.json"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load class mapping
        with open(class_mapping_path, "r") as f:
            self.idx_to_class = json.load(f)
        self.num_classes = len(self.idx_to_class)

        # Define model structure
        self.model = resnet50()
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes)
        )
        
        # Load weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model = self.model.to(self.device)
        self.model.eval()

        # Preprocessing transform (handles arbitrary user inputs)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path_or_file):
        # Load image
        if isinstance(image_path_or_file, str):
            img = Image.open(image_path_or_file)
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
        else:
            img = Image.open(image_path_or_file)
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")

        # Apply transform and add batch dimension
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        # Get top 5 classes
        top_prob, top_idx = torch.topk(probabilities, min(5, self.num_classes))
        
        top_5_results = []
        for prob, idx in zip(top_prob, top_idx):
            class_name = self.idx_to_class[str(idx.item())]
            top_5_results.append({
                "class": class_name,
                "confidence": float(prob.item())
            })

        return {
            "prediction": top_5_results[0]["class"],
            "confidence": top_5_results[0]["confidence"],
            "top_5": top_5_results
        }
