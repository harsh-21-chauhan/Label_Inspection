import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_transforms(split="train"):
    """
    Standard transforms for ShuffleNet V2 (expects 224x224 RGB images).
    Adds slight illumination jitter for train split to improve generalization.
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_defect_dataloader(data_dir="data/defect_dataset", split="train", batch_size=16, shuffle=True):
    """
    Loads defect classification dataset structured as:
      0_normal/    -> Label 0 (Defect-Free)
      1_defective/ -> Label 1 (Defective: Missing/Blur/Adhesion)
    """
    split_dir = os.path.join(data_dir, split)
    if not os.path.exists(split_dir):
        return None, None
    transform = get_transforms(split)
    dataset = datasets.ImageFolder(split_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return loader, dataset

if __name__ == "__main__":
    loader, ds = get_defect_dataloader()
    if ds is not None:
        print(f"Loaded {len(ds)} samples across {len(ds.classes)} classes: {ds.class_to_idx}")
    else:
        print("Defect dataset directory not found. Run generate_defect_dataset.py first.")
