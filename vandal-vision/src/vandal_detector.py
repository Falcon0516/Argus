"""
vandal_detector.py
==================
Real-time Vandalism (Spitting & Graffiti) Detection engine.

Adapted from: Arr0w28/Off_VandalVision (Vindhler)
Uses a custom PyTorch CNN to classify frames.
"""
from __future__ import annotations

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from datetime import datetime


class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = torch.nn.Linear(32 * 56 * 56, 128)
        self.fc2 = torch.nn.Linear(128, 3)  # 3 classes: spitting, graffiti, no graffiti

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


CLASS_NAMES = {
    0: "SPITTING",
    1: "GRAFFITI",
    2: "SAFE"
}

CLASS_COLORS = {
    0: (0, 165, 255),   # Orange for spitting
    1: (0, 0, 255),     # Red for graffiti
    2: (0, 255, 0)      # Green for safe
}


class VandalDetector:
    def __init__(self, model_path: str = "models/vandalism.pth") -> None:
        self.model = SimpleNet()
        
        # Determine device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        # Load weights safely
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb_frame)
        
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            
            # Apply softmax to get confidence scores
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf_val, predicted = torch.max(probs.data, 1)
            
            label_idx = predicted.item()
            conf = conf_val.item()

        annotated = frame.copy()
        events = []
        
        if label_idx in [0, 1] and conf > 0.60:
            cls_name = CLASS_NAMES[label_idx]
            color = CLASS_COLORS[label_idx]
            
            # Draw prominent border around frame
            h, w = annotated.shape[:2]
            cv2.rectangle(annotated, (0, 0), (w-1, h-1), color, 8)
            
            # Top-left warning label
            label = f"⚠ VANDALISM: {cls_name} ({conf:.2f})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.8, 1)
            cv2.rectangle(annotated, (10, 10), (10 + tw + 20, 10 + th + 20), color, -1)
            cv2.putText(annotated, label, (20, 10 + th + 10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
            
            events.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "type": cls_name,
                "conf": round(conf, 2)
            })

        return annotated, events
