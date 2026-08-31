import cv2
import numpy as np
import os

def align_image_with_sift(template_path, target_path, output_path):
    """
    Part 2: Data Creation - SIFT Image Matching
    Implementation of Section 3.3.4 of the paper.
    Calculates the affine transformation to align a target image 
    to a template image using SIFT features.
    """
    # 1. Read images in grayscale
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
    
    if template is None or target is None:
        print("Error: Could not read one or both images.")
        return None

    # 2. Extract SIFT features
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(target, None)

    # 3. Feature matching using FLANN-based k-nearest neighbor
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        print("Not enough descriptors found.")
        return None
        
    matches = flann.knnMatch(des1, des2, k=2)

    # 4. Filter matches using Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    print(f"Found {len(good_matches)} good matches.")

    # 5. Calculate Transformation Matrix if enough matches found
    MIN_MATCH_COUNT = 4 # Affine needs at least 3, Homography needs 4
    if len(good_matches) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Using findHomography (or estimateAffinePartial2D) to find transformation
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        # 6. Apply affine transformation to align target to template
        h, w = template.shape
        aligned_image = cv2.warpPerspective(target, M, (w, h))
        
        cv2.imwrite(output_path, aligned_image)
        print(f"Successfully aligned and saved to {output_path}")
        return aligned_image
    else:
        print(f"Not enough matches are found - {len(good_matches)}/{MIN_MATCH_COUNT}")
        return None

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    print("Testing SIFT Alignment...")
    align_image_with_sift("data/bottle1.png", "data/bottle2.jpg", "outputs/aligned_bottle2.jpg")
