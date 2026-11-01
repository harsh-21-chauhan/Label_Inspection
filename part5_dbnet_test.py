import os
import json
import torch
import cv2
import numpy as np
from part5_dbnet_model import DBNet


def test_dbnet(num_test_samples=15):
    """
    Phase 5: DBNet Testing & Evaluation Script
    Loads the trained model and runs inference on unseen test images from curated_dataset.
    Draws the detected text regions, calculates IoU, and outputs heatmaps.
    """
    print("=" * 60)
    print("Phase 5: DBNet Test & Evaluation on Curated Test Set")
    print("=" * 60)

    model_path = "outputs/dbnet_trained.pth"
    test_img_dir = "data/curated_dataset/test_images"
    test_ann_path = "data/curated_dataset/annotations/test_bottles_curated.json"
    output_dir = "outputs/dbnet_test_results"
    os.makedirs(output_dir, exist_ok=True)

    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load trained model
    model = DBNet().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded trained model from {model_path} onto {device}")
    else:
        print(f"WARNING: No trained model found at {model_path}. Using random weights.")
    model.eval()

    # Load test annotations
    with open(test_ann_path, "r") as f:
        test_data = json.load(f)

    test_images = test_data["images"][:num_test_samples]
    total_iou = 0.0
    valid_count = 0

    print(f"\nRunning inference on {len(test_images)} unseen test bottle/label images...\n")

    for img_info in test_images:
        file_name = img_info["file_name"]
        img_path = os.path.join(test_img_dir, file_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        orig_h, orig_w = img.shape[:2]

        # Preprocess: resize to 640x640 and normalize
        img_resized = cv2.resize(img, (640, 640))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # Forward pass
        with torch.no_grad():
            prob_map, thresh_map, binary_map = model(img_tensor)

        # Probability map to numpy
        prob_np = prob_map.squeeze().cpu().numpy()

        # Standard DBNet post-processing: adaptive thresholding + smoothing + morphological closing
        p_smooth = cv2.GaussianBlur(prob_np, (5, 5), 0)
        th = max(0.40, float(p_smooth.mean() + 0.8 * p_smooth.std()))
        text_mask_640 = (p_smooth > th).astype(np.uint8) * 255

        # Morphological closing to group adjacent character strokes into text lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
        closed_mask = cv2.morphologyEx(text_mask_640, cv2.MORPH_CLOSE, kernel)
        text_mask_orig = cv2.resize(closed_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # Find external contours and filter tiny speckle noise (< 80 pixels)
        all_contours, _ = cv2.findContours(text_mask_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in all_contours if cv2.contourArea(c) > 80]

        # Draw detected regions on original image
        result_img = img.copy()
        cv2.drawContours(result_img, contours, -1, (0, 255, 0), 2)

        # Ground truth mask for IoU calculation
        gt_anns = [a for a in test_data["annotations"] if a["image_id"] == img_info["id"]]
        gt_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for ann in gt_anns:
            if "bezier_pts" in ann and ann["bezier_pts"]:
                raw_pts = ann["bezier_pts"]
            elif "segmentation" in ann and len(ann["segmentation"]) > 0 and len(ann["segmentation"][0]) >= 6:
                raw_pts = ann["segmentation"][0]
            elif "bbox" in ann and len(ann["bbox"]) == 4:
                bx, by, bw, bh = ann["bbox"]
                raw_pts = [bx, by, bx + bw, by, bx + bw, by + bh, bx, by + bh]
            else:
                continue
            pts = np.array(raw_pts, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(gt_mask, [pts], 255)

        # IoU calculation
        intersection = np.logical_and(text_mask_orig > 0, gt_mask > 0).sum()
        union = np.logical_or(text_mask_orig > 0, gt_mask > 0).sum()
        iou = intersection / (union + 1e-6)
        total_iou += iou
        valid_count += 1

        # Save result image
        save_path = os.path.join(output_dir, f"detected_{file_name}")
        cv2.imwrite(save_path, result_img)

        # Save probability heatmap overlay
        heatmap = cv2.applyColorMap((prob_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h))
        overlay = cv2.addWeighted(img, 0.6, heatmap_resized, 0.4, 0)
        cv2.imwrite(os.path.join(output_dir, f"heatmap_{file_name}"), overlay)

        print(f"  {file_name:30s} | Detected {len(contours):2d} text regions | IoU: {iou:.4f}")

    mean_iou = total_iou / max(valid_count, 1)
    print(f"\n{'=' * 60}")
    print(f"EVALUATION COMPLETE:")
    print(f"  Images Evaluated: {valid_count}")
    print(f"  Mean IoU:         {mean_iou:.4f}")
    print(f"  Visual Results:   {output_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    test_dbnet()
