import os
import json
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

class CTW1500Dataset(Dataset):
    """
    Phase 4: Text Detection Dataset
    Loads CTW1500 images and our synthesized defective images.
    Generates segmentation masks from the curved bezier points
    so the Mask R-CNN model can learn to detect curved text.
    """
    def __init__(self, json_path, img_dirs):
        self.img_dirs = img_dirs
        
        with open(json_path, 'r') as f:
            self.coco_data = json.load(f)
            
        # We only use a tiny subset (first 5 images) for demonstration
        self.images = self.coco_data['images'][:5]
        self.annotations = self.coco_data['annotations']

    def __len__(self):
        # Multiply by 3 because we have Original, Blur, and Missing for each image
        return len(self.images) * 3

    def __getitem__(self, idx):
        # Determine if we are loading Original (0), Blur (1), or Missing (2)
        variant = idx % 3
        img_info = self.images[idx // 3]
        
        img_id = img_info['id']
        file_name = img_info['file_name']
        
        if variant == 0:
            img_path = os.path.join(self.img_dirs['original'], file_name)
        elif variant == 1:
            img_path = os.path.join(self.img_dirs['augmented'], f"blur_{file_name}")
        else:
            img_path = os.path.join(self.img_dirs['augmented'], f"missing_{file_name}")
            
        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = F.to_tensor(img) # Convert to PyTorch Tensor (C, H, W)
        
        # Get annotations
        anns = [a for a in self.annotations if a['image_id'] == img_id]
        
        num_objs = len(anns)
        boxes = []
        masks = np.zeros((num_objs, img.shape[1], img.shape[2]), dtype=np.uint8)
        labels = torch.ones((num_objs,), dtype=torch.int64) # Label 1 means 'Text'
        
        for i, ann in enumerate(anns):
            # Bounding Box (xmin, ymin, width, height) -> (xmin, ymin, xmax, ymax)
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            
            # Mask (from curved bezier points)
            pts = np.array(ann['bezier_pts'], np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(masks[i], [pts], 1)
            
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        
        return img, target

# Helper function to handle variable sized inputs in dataloader
def collate_fn(batch):
    return tuple(zip(*batch))
