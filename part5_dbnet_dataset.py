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
    def __init__(self, ann_path, img_dir, augmented_dir, target_size=640):
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
        # Original + Blur + Missing for each image
        return len(self.images) * 3

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
        variant = idx % 3   # 0=original, 1=blur, 2=missing
        img_info = self.images[idx // 3]
        img_id = img_info['id']
        file_name = img_info['file_name']

        # Choose image path based on variant
        if variant == 0:
            img_path = os.path.join(self.img_dir, file_name)
        elif variant == 1:
            aug_path = os.path.join(self.augmented_dir, f"blur_{file_name}")
            img_path = aug_path if os.path.exists(aug_path) else os.path.join(self.img_dir, file_name)
        else:
            aug_path = os.path.join(self.augmented_dir, f"missing_{file_name}")
            img_path = aug_path if os.path.exists(aug_path) else os.path.join(self.img_dir, file_name)

        # Load and resize image
        img = cv2.imread(img_path)
        if img is None:
            # Fallback to original if augmented doesn't exist
            img = cv2.imread(os.path.join(self.img_dir, file_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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
            # Get bezier points and scale them
            pts = np.array(ann['bezier_pts'], np.float32).reshape((-1, 2))
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
    # Quick test
    dataset = DBNetDataset(
        ann_path="data/CTW1500/annotations/train_ctw1500_maxlen100_v2.json",
        img_dir="data/CTW1500/ctwtrain_text_image",
        augmented_dir="outputs/ctw1500_augmented/train",
        target_size=640
    )
    print(f"Dataset size: {len(dataset)} images (1000 originals x 3 variants)")
    img, prob, thresh, tmask = dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Probability map shape: {prob.shape}")
    print(f"Threshold map shape: {thresh.shape}")
    print(f"Text pixels in prob map: {(prob > 0).sum().item()}")
