import cv2
import json
import numpy as np
import os
import random
import copy

def apply_masked_defect(image, mask, defect_type):
    """
    Applies a morphological defect only to the masked area (the text).
    This simulates industrial printing errors without ruining the natural background!
    """
    if defect_type == 'missing':
        # Heavy dilation makes dark text disappear
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        defective_img = cv2.dilate(image, kernel, iterations=2)
    elif defect_type == 'blur':
        # Light dilation fades the text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        defective_img = cv2.dilate(image, kernel, iterations=1)
    else:
        return image
        
    # Blend: Use defective image where mask is > 0, else use original image
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    blended = np.where(mask_3ch > 0, defective_img, image)
    return blended

def augment_ctw1500():
    """
    Phase 3: Data Synthesis Pipeline
    Reads CTW1500 annotations, masks the curved text polygons, 
    and applies synthetic defects to make the Text Detector robust 
    to industrial printing errors.
    """
    print("--- Starting Phase 3: CTW1500 Targeted Defect Synthesis ---")
    
    img_dir = "data/CTW1500/ctwtrain_text_image"
    ann_path = "data/CTW1500/annotations/train_ctw1500_maxlen100_v2.json"
    out_dir = "outputs/ctw1500_augmented"
    out_ann_path = "outputs/ctw1500_augmented/train_augmented.json"
    
    if not os.path.exists(img_dir) or not os.path.exists(ann_path):
        print("Error: CTW1500 dataset not found in data/CTW1500/")
        return
        
    os.makedirs(out_dir, exist_ok=True)
    
    with open(ann_path, 'r') as f:
        coco_data = json.load(f)
        
    new_coco_data = copy.deepcopy(coco_data)
    
    # Process only the first 5 images for demonstration
    img_list = coco_data['images'][:5]
    
    for img_info in img_list:
        img_id = img_info['id']
        file_name = img_info['file_name']
        img_path = os.path.join(img_dir, file_name)
        
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        # Get all annotations for this image
        anns = [a for a in coco_data['annotations'] if a['image_id'] == img_id]
        
        # Create a black mask
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Draw all curved text polygons onto the mask
        for ann in anns:
            pts = np.array(ann['bezier_pts'], np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)
            
        # Synthesize a 'blur' defect
        blurred_image = apply_masked_defect(image, mask, 'blur')
        blur_filename = f"blur_{file_name}"
        cv2.imwrite(os.path.join(out_dir, blur_filename), blurred_image)
        
        # Synthesize a 'missing' defect
        missing_image = apply_masked_defect(image, mask, 'missing')
        missing_filename = f"missing_{file_name}"
        cv2.imwrite(os.path.join(out_dir, missing_filename), missing_image)
        
        print(f"Processed {file_name} -> Generated robust variants.")

    print(f"\nPhase 3 Synthesis Complete! Augmented images saved to {out_dir}")

if __name__ == "__main__":
    augment_ctw1500()
