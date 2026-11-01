import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import pyclipper
from shapely.geometry import Polygon


class DBNetDataset(Dataset):
    """
    Phase 5: DBNet Dataset Loader
    Loads CTW1500 images and generates the special "shrink maps" that DBNet
    needs for training. DBNet does NOT use bounding boxes like Mask R-CNN.
    Instead, it learns from pixel-level probability maps.

    For each image, we generate:
    1. Probability Map (GT): A binary mask where text pixels = 1, background = 0
    2. Threshold Map (GT): A border region around each text polygon
    """
    def __init__(self, ann_path, img_dir, augmented_dir=None, target_size=640):
        self.img_dir = img_dir
        self.augmented_dir = augmented_dir
        self.target_size = target_size

        with open(ann_path, 'r') as f:
            self.coco_data = json.load(f)

        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']

        # Build a mapping: image_id -> list of annotations
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

    def __len__(self):
        # 3 dynamic variants for each bottle: 0=clean, 1=conveyor/glare blur, 2=morphological ink defect
        return len(self.images) * 3

    def _apply_industrial_augmentation(self, img, variant):
        """
        On-The-Fly Industrial Data Augmentation tailored specifically for bottle labels:
        
        Variant 0: Clean baseline image (learns uncorrupted label geometry)
        Variant 1: Optical Conveyor Perturbations:
            - Photometric jitter (brightness / contrast shifts simulating strobe reflections on glass/PET)
            - Motion / Defocus blur (conveyor vibration)
        Variant 2: Morphological Ink Degradations (from Section 3.3.3 of the paper):
            - Morphological dilation (simulates ink bleed / character adhesion) OR
            - Morphological erosion (simulates low ink / broken character strokes)
            - Ambient factory lighting shifts
        """
        if variant == 0:
            return img

        aug = img.copy()

        if variant == 1:
            # 1. Factory Lighting & Glare Jitter (contrast & brightness)
            alpha = np.random.uniform(0.80, 1.25)
            beta = np.random.uniform(-18, 18)
            aug = np.clip(aug.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

            # 2. Conveyor Motion Blur / Camera Defocus
            ksize = int(np.random.choice([3, 5]))
            aug = cv2.GaussianBlur(aug, (ksize, ksize), sigmaX=float(np.random.uniform(0.6, 1.4)))

        elif variant == 2:
            # 3. Morphological Ink Degradation (Paper Section 3.3.3)
            op = np.random.choice(['dilate', 'erode', 'blur'])
            if op == 'dilate':
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                aug = cv2.dilate(aug, kernel, iterations=1)
            elif op == 'erode':
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                aug = cv2.erode(aug, kernel, iterations=1)
            else:
                aug = cv2.medianBlur(aug, 3)

            alpha = np.random.uniform(0.85, 1.15)
            aug = np.clip(aug.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

        return aug

    def _shrink_polygon(self, polygon, shrink_ratio=0.4):
        """
        Shrinks a polygon inward by a ratio. This creates the "shrunk" ground truth
        that DBNet uses. The idea: the CENTER of the text is definitely text,
        but the EDGES are ambiguous. So we train on slightly shrunk polygons.
        """
        poly = Polygon(polygon)
        if not poly.is_valid or poly.area < 1:
            return polygon
        distance = poly.area * (1 - shrink_ratio ** 2) / (poly.length + 1e-6)
        subject = [tuple(p) for p in polygon]
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        shrunk = pco.Execute(-distance)
        if len(shrunk) == 0:
            return polygon
        return np.array(shrunk[0])

    def __getitem__(self, idx):
        variant = idx % 3   # 0=clean, 1=conveyor/glare blur, 2=morphological ink defect
        img_info = self.images[idx // 3]
        img_id = img_info['id']
        file_name = img_info['file_name']

        # Load original image
        img_path = os.path.join(self.img_dir, file_name)
        img = cv2.imread(img_path)
        if img is None:
            # If image missing, return zero dummy tensor
            img = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply dynamic industrial augmentation on-the-fly
        img = self._apply_industrial_augmentation(img, variant)

        orig_h, orig_w = img.shape[:2]
        img = cv2.resize(img, (self.target_size, self.target_size))

        # Scale factors for resizing annotations
        sx = self.target_size / orig_w
        sy = self.target_size / orig_h

        # Generate ground truth maps
        prob_map = np.zeros((self.target_size, self.target_size), dtype=np.float32)
        thresh_map = np.zeros((self.target_size, self.target_size), dtype=np.float32)
        thresh_mask = np.zeros((self.target_size, self.target_size), dtype=np.float32)

        anns = self.img_to_anns.get(img_id, [])

        for ann in anns:
            if 'bezier_pts' in ann and ann['bezier_pts']:
                raw_pts = ann['bezier_pts']
            elif 'segmentation' in ann and len(ann['segmentation']) > 0 and len(ann['segmentation'][0]) >= 6:
                raw_pts = ann['segmentation'][0]
            elif 'bbox' in ann and len(ann['bbox']) == 4:
                bx, by, bw, bh = ann['bbox']
                raw_pts = [bx, by, bx + bw, by, bx + bw, by + bh, bx, by + bh]
            else:
                continue

            pts = np.array(raw_pts, np.float32).reshape((-1, 2))
            pts[:, 0] *= sx
            pts[:, 1] *= sy
            pts = pts.astype(np.int32)

            if len(pts) < 3:
                continue

            # Fill the full polygon on the probability map
            cv2.fillPoly(prob_map, [pts], 1.0)

            # Generate shrunk polygon for the threshold map border region
            try:
                shrunk = self._shrink_polygon(pts, shrink_ratio=0.4)
                if len(shrunk) >= 3:
                    shrunk = shrunk.astype(np.int32)
                    # The threshold map captures the BORDER between shrunk and full polygon
                    cv2.fillPoly(thresh_mask, [pts], 1.0)
                    cv2.fillPoly(thresh_map, [pts], 0.7)
                    cv2.fillPoly(thresh_map, [shrunk], 0.3)
            except Exception:
                pass

        # Convert to tensors
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        prob_tensor = torch.from_numpy(prob_map).unsqueeze(0)
        thresh_tensor = torch.from_numpy(thresh_map).unsqueeze(0)
        thresh_mask_tensor = torch.from_numpy(thresh_mask).unsqueeze(0)

        return img_tensor, prob_tensor, thresh_tensor, thresh_mask_tensor


if __name__ == "__main__":
    # Quick test on merged curated dataset
    dataset = DBNetDataset(
        ann_path="data/curated_dataset/annotations/train_merged.json",
        img_dir="data/curated_dataset/train_images",
        augmented_dir="outputs/augmented/train",
        target_size=640
    )
    print(f"Dataset size: {len(dataset)} items ({len(dataset.images)} originals x 3 variants)")
    img, prob, thresh, tmask = dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Probability map shape: {prob.shape}")
    print(f"Threshold map shape: {thresh.shape}")
    print(f"Text pixels in prob map: {(prob > 0).sum().item()}")

    # Test an image from each origin (CTW, Total-Text, Scraped)
    print("Testing index 0 (CTW):", dataset[0][1].sum().item() > 0)
    print("Testing index 400 (TT):", dataset[400][1].sum().item() > 0)
    print("Testing index 900 (Scraped):", dataset[900][1].sum().item() > 0)
