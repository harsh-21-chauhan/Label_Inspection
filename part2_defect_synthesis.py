import cv2
import numpy as np
import os

def synthesize_defects(image_path, output_dir):
    """
    Part 2: Data Creation - Morphological Processing
    Implementation of Section 3.3.3 of the paper.
    Synthesizes defective code images to balance the dataset 
    for deep learning training.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(image_path).split('.')[0]
    
    print(f"Synthesizing defects for {base_name}...")

    # The paper mentions that codes are black on a shallow background.
    # Therefore, DILATION will expand the background (white), shrinking the black codes.
    # EROSION will shrink the background (white), expanding the black codes (making them thicker/adhered).
    
    # 1. Simulate Code Missing (Heavy Dilation)
    # Using a large rectangular structuring element to completely eliminate parts of the code
    kernel_missing = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    code_missing = cv2.dilate(image, kernel_missing, iterations=2)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_defect_missing.jpg"), code_missing)
    
    # 2. Simulate Code Blur (Light Dilation)
    # Using a smaller structuring element so the code color becomes lighter and blurred
    kernel_blur = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    code_blur = cv2.dilate(image, kernel_blur, iterations=1)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_defect_blur.jpg"), code_blur)
    
    # 3. Simulate Code Adhesion (Opening / Erosion)
    # The paper states: "By using a rectangular structure element to process the opening operation, 
    # a defective image with the code adhesion can be obtained."
    # Opening is erosion followed by dilation.
    kernel_adhesion = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    code_adhesion = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_adhesion)
    
    # Alternatively, just a direct erosion also heavily thickens the black dot-matrix characters
    code_adhesion_alt = cv2.erode(image, kernel_adhesion, iterations=1)
    
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_defect_adhesion.jpg"), code_adhesion)
    
    print(f"Successfully generated 3 defective variants in {output_dir}/")

if __name__ == "__main__":
    # Test on a base image from curated product dataset
    sample_img = "data/CTW1500_filtered/train_images/0292.jpg"
    synthesize_defects(sample_img, "outputs/synthesized_defects")
