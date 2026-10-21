import cv2
import json
import numpy as np
import os
import copy

def apply_masked_defect(image, mask, defect_type):
    """
    Applies a morphological defect only to the masked area (the text).
    This simulates industrial printing errors without ruining the natural background.
    """
    if defect_type == 'missing':
        # Heavy dilation makes dark text disappear (simulates clogged nozzle)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        defective_img = cv2.dilate(image, kernel, iterations=2)
    elif defect_type == 'blur':
        # Light dilation fades the text (simulates ink spread)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        defective_img = cv2.dilate(image, kernel, iterations=1)
    else:
        return image

    # Blend: Use defective image where mask is > 0, else use original image
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    blended = np.where(mask_3ch > 0, defective_img, image)
    return blended


def augment_split(img_dir, ann_path, out_dir, split_name, max_images=None):
    """
    Processes a full split (train or test) of CTW1500.
    Generates blur and missing-ink variants for each image.
    """
    if not os.path.exists(img_dir) or not os.path.exists(ann_path):
        print(f"Error: {split_name} data not found.")
        return 0

    os.makedirs(out_dir, exist_ok=True)

    with open(ann_path, 'r') as f:
        coco_data = json.load(f)

    img_list = coco_data['images']
    if max_images:
        img_list = img_list[:max_images]

    processed = 0
    for img_info in img_list:
        img_id = img_info['id']
        file_name = img_info['file_name']
        img_path = os.path.join(img_dir, file_name)

        image = cv2.imread(img_path)
        if image is None:
            continue

        # Get all annotations for this image
        anns = [a for a in coco_data['annotations'] if a['image_id'] == img_id]
        if len(anns) == 0:
            continue

        # Create a black mask
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Draw all curved text polygons onto the mask
        for ann in anns:
            pts = np.array(ann['bezier_pts'], np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)

        # Save original copy to augmented folder (so all images are in one place)
        cv2.imwrite(os.path.join(out_dir, file_name), image)

        # Synthesize a 'blur' defect
        blurred_image = apply_masked_defect(image, mask, 'blur')
        cv2.imwrite(os.path.join(out_dir, f"blur_{file_name}"), blurred_image)

        # Synthesize a 'missing' defect
        missing_image = apply_masked_defect(image, mask, 'missing')
        cv2.imwrite(os.path.join(out_dir, f"missing_{file_name}"), missing_image)

        processed += 1
        if processed % 50 == 0:
            print(f"  [{split_name}] Processed {processed}/{len(img_list)} images...")

    print(f"  [{split_name}] Done! {processed} images -> {processed * 3} total (original + blur + missing)")
    return processed


def augment_ctw1500():
    """
    Phase 3 (Updated): Full CTW1500 Augmentation Pipeline
    Processes both train and test splits completely.
    """
    print("=" * 60)
    print("Phase 3: Full CTW1500 Defect Synthesis Pipeline")
    print("=" * 60)

    # Process training split
    print("\n[1/2] Processing TRAINING split...")
    train_count = augment_split(
        img_dir="data/CTW1500/ctwtrain_text_image",
        ann_path="data/CTW1500/annotations/train_ctw1500_maxlen100_v2.json",
        out_dir="outputs/ctw1500_augmented/train",
        split_name="TRAIN"
    )

    # Process test split
    print("\n[2/2] Processing TEST split...")
    test_count = augment_split(
        img_dir="data/CTW1500/ctwtest_text_image",
        ann_path="data/CTW1500/annotations/test_ctw1500_maxlen100.json",
        out_dir="outputs/ctw1500_augmented/test",
        split_name="TEST"
    )

    print(f"\n{'=' * 60}")
    print(f"COMPLETE: {train_count + test_count} base images -> {(train_count + test_count) * 3} augmented images")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    augment_ctw1500()
