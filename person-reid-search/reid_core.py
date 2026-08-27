import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import torchreid

class ReidFeatureExtractor:
    def __init__(self, model_name='osnet_x0_25', device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load lightweight OSNet model pre-trained on Market1501/MSMT17
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1000,
            loss='softmax',
            pretrained=True
        )
        self.model.to(self.device)
        self.model.eval()

        # Standard ReID input transforms (Height: 256, Width: 128)
        self.transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract(self, image_np):
        """Extracts a 512-dim normalized feature vector from a cropped BGR image."""
        img_rgb = Image.fromarray(image_np[:, :, ::-1])
        tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(tensor)
            # L2 Normalize the vector for cosine distance computation
            features = features / torch.norm(features, p=2, dim=1, keepdim=True)
            
        return features.cpu().numpy().flatten()