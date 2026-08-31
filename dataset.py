import os
import glob
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class LabelDefectDataset(Dataset):
    """
    Custom PyTorch Dataset for loading bottle labels.
    We classify images into two categories:
    0: Defect-Free (Good)
    1: Defective (Bad)
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load Defect-Free Images (Label 0)
        # Using the base template and aligned target as our "good" samples
        good_samples = glob.glob("data/*.png") + glob.glob("outputs/aligned_*.jpg")
        for img_path in good_samples:
            self.images.append(img_path)
            self.labels.append(0) # 0 = Defect-Free
            
        # Load Defective Images (Label 1)
        # Using the synthesized missing, blurred, and adhered codes
        bad_samples = glob.glob("outputs/synthesized_defects/*.jpg")
        for img_path in bad_samples:
            self.images.append(img_path)
            self.labels.append(1) # 1 = Defective
            
        print(f"Loaded {len(self.images)} total images for training.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Read image using PIL (standard for torchvision transforms)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Standard transforms for ShuffleNet V2 (expects 224x224 RGB images)
get_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if __name__ == "__main__":
    # Quick test to ensure dataset loads correctly
    dataset = LabelDefectDataset(".", transform=get_transforms)
    if len(dataset) > 0:
        img, label = dataset[0]
        print(f"Sample Image Shape: {img.shape}, Label: {label}")
