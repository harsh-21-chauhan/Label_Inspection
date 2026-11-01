import os
import json
import cv2
import numpy as np
from PIL import Image

def generate_shufflenet_defect_dataset(
    ann_path="data/curated_dataset/annotations/train_bottles_curated.json",
    img_dir="data/curated_dataset/train_images",
    output_base="data/defect_dataset",
    max_crops=300
):
    """
    Crops genuine label text regions from the curated bottle dataset,
    then applies Section 3.3.3 morphological operations to synthesize:
      - 0_normal: Crisp, uncorrupted label text
      - 1_defective:
          * Blurred Code (light dilation)
          * Missing Code (heavy dilation)
          * Code Adhesion (morphological opening/erosion)
    """
    print("=" * 60)
    print("Generating Industrial Defect Dataset for ShuffleNet V2")
    print("=" * 60)

    train_normal = os.path.join(output_base, "train", "0_normal")
    train_defect = os.path.join(output_base, "train", "1_defective")
    val_normal = os.path.join(output_base, "val", "0_normal")
    val_defect = os.path.join(output_base, "val", "1_defective")

    for p in [train_normal, train_defect, val_normal, val_defect]:
        os.makedirs(p, exist_ok=True)

    with open(ann_path, "r") as f:
        coco_data = json.load(f)

    img_map = {im["id"]: im["file_name"] for im in coco_data["images"]}
    anns = coco_data["annotations"]
    np.random.seed(42)
    np.random.shuffle(anns)

    crop_count = 0
    k_missing = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    k_blur = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k_adhesion = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))

    for ann in anns:
        if crop_count >= max_crops:
            break

        img_id = ann["image_id"]
        if img_id not in img_map:
            continue
        fname = img_map[img_id]
        img_path = os.path.join(img_dir, fname)
        if not os.path.exists(img_path):
            continue

        # Get bounding box
        if "bbox" in ann and len(ann["bbox"]) == 4:
            bx, by, bw, bh = [int(v) for v in ann["bbox"]]
        elif "segmentation" in ann and ann["segmentation"] and len(ann["segmentation"][0]) >= 6:
            pts = np.array(ann["segmentation"][0]).reshape((-1, 2))
            bx, by, bw, bh = cv2.boundingRect(pts.astype(np.int32))
        elif "bezier_pts" in ann and ann["bezier_pts"]:
            pts = np.array(ann["bezier_pts"]).reshape((-1, 2))
            bx, by, bw, bh = cv2.boundingRect(pts.astype(np.int32))
        else:
            continue

        if bw < 25 or bh < 12:
            continue

        mat = cv2.imread(img_path)
        if mat is None:
            continue
        h, w = mat.shape[:2]

        # Pad slightly around text region
        pad_x = int(bw * 0.1)
        pad_y = int(bh * 0.1)
        x1 = max(0, bx - pad_x)
        y1 = max(0, by - pad_y)
        x2 = min(w, bx + bw + pad_x)
        y2 = min(h, by + bh + pad_y)

        crop = mat[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)

        # 80% train, 20% val
        split = "train" if (crop_count % 5 != 0) else "val"
        norm_dir = train_normal if split == "train" else val_normal
        def_dir = train_defect if split == "train" else val_defect

        # 1. Save Clean / Normal sample (Class 0)
        base_name = f"crop_{crop_count:04d}"
        cv2.imwrite(os.path.join(norm_dir, f"{base_name}_normal.jpg"), crop_resized)

        # 2. Synthesize Defect: Missing Code (Section 3.3.3 heavy dilation)
        defect_missing = cv2.dilate(crop_resized, k_missing, iterations=1)
        cv2.imwrite(os.path.join(def_dir, f"{base_name}_def_missing.jpg"), defect_missing)

        # 3. Synthesize Defect: Blurred Code (Section 3.3.3 light dilation + gaussian blur)
        defect_blur = cv2.dilate(crop_resized, k_blur, iterations=1)
        defect_blur = cv2.GaussianBlur(defect_blur, (5, 5), 1.2)
        cv2.imwrite(os.path.join(def_dir, f"{base_name}_def_blur.jpg"), defect_blur)

        # 4. Synthesize Defect: Code Adhesion (Section 3.3.3 morphological opening / erosion)
        defect_adhesion = cv2.erode(crop_resized, k_adhesion, iterations=1)
        cv2.imwrite(os.path.join(def_dir, f"{base_name}_def_adhesion.jpg"), defect_adhesion)

        crop_count += 1

    train_n = len(os.listdir(train_normal))
    train_d = len(os.listdir(train_defect))
    val_n = len(os.listdir(val_normal))
    val_d = len(os.listdir(val_defect))

    print(f"Dataset generated successfully in {output_base}/:")
    print(f"  Train: {train_n} normal crops, {train_d} defective crops (Total: {train_n + train_d})")
    print(f"  Val:   {val_n} normal crops, {val_d} defective crops (Total: {val_n + val_d})")
    print("=" * 60)

if __name__ == "__main__":
    generate_shufflenet_defect_dataset()
