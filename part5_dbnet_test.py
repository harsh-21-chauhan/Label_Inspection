import torch
import cv2
import numpy as np
import os
import json
from part5_dbnet_model import DBNet


def test_dbnet():
    """
    Phase 5: DBNet Testing & Evaluation Script
    Loads the trained model and runs inference on unseen test images.
    Draws the detected text regions and calculates accuracy metrics.
    """
    print("=" * 60)
    print("Phase 5: DBNet Test & Evaluation")
    print("=" * 60)

    model_path = "outputs/dbnet_trained.pth"
    test_img_dir = "data/CTW1500/ctwtest_text_image"
    test_ann_path = "data/CTW1500/annotations/test_ctw1500_maxlen100.json"
    output_dir = "outputs/dbnet_test_results"
    os.makedirs(output_dir, exist_ok=True)

    # Load trained model
    model = DBNet()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"Loaded trained model from {model_path}")
    else:
        print(f"WARNING: No trained model found at {model_path}. Using random weights for demo.")
    model.eval()

    # Load test annotations for evaluation
    with open(test_ann_path, 'r') as f:
        test_data = json.load(f)

    # Test on first 10 images
    test_images = test_data['images'][:10]

    total_iou = 0.0
    valid_count = 0

    print(f"\nRunning inference on {len(test_images)} test images...\n")

    for img_info in test_images:
        file_name = img_info['file_name']
        img_path = os.path.join(test_img_dir, file_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        orig_h, orig_w = img.shape[:2]

        # Preprocess: resize to 640x640 and normalize
        img_resized = cv2.resize(img, (640, 640))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

        # Forward pass (no gradient needed for inference)
        with torch.no_grad():
            prob_map, thresh_map, binary_map = model(img_tensor)

        # Convert probability map to numpy for visualization
        prob_np = prob_map.squeeze().cpu().numpy()

        # Threshold the probability map to get text regions
        text_mask = (prob_np > 0.3).astype(np.uint8) * 255

        # Resize back to original image size
        text_mask_orig = cv2.resize(text_mask, (orig_w, orig_h))

        # Find contours (detected text regions)
        contours, _ = cv2.findContours(text_mask_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw detected regions on original image
        result_img = img.copy()
        cv2.drawContours(result_img, contours, -1, (0, 255, 0), 2)

        # Calculate IoU against ground truth
        gt_anns = [a for a in test_data['annotations'] if a['image_id'] == img_info['id']]
        gt_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for ann in gt_anns:
            pts = np.array(ann['bezier_pts'], np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(gt_mask, [pts], 255)

        # IoU = Intersection / Union
        intersection = np.logical_and(text_mask_orig > 0, gt_mask > 0).sum()
        union = np.logical_or(text_mask_orig > 0, gt_mask > 0).sum()
        iou = intersection / (union + 1e-6)
        total_iou += iou
        valid_count += 1

        # Save visualization
        save_path = os.path.join(output_dir, f"detected_{file_name}")
        cv2.imwrite(save_path, result_img)

        # Also save the probability heatmap
        heatmap = cv2.applyColorMap((prob_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h))
        overlay = cv2.addWeighted(img, 0.6, heatmap_resized, 0.4, 0)
        cv2.imwrite(os.path.join(output_dir, f"heatmap_{file_name}"), overlay)

        print(f"  {file_name}: Detected {len(contours)} text regions | IoU: {iou:.4f}")

    # Final metrics
    mean_iou = total_iou / max(valid_count, 1)
    print(f"\n{'=' * 60}")
    print(f"RESULTS: Mean IoU = {mean_iou:.4f} over {valid_count} test images")
    print(f"Visualizations saved to {output_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    test_dbnet()
