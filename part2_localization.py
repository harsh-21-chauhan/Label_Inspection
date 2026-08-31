import cv2
import numpy as np

def localize_text(image_path, output_path):
    """
    Part 2: Text Localization (Region of Interest Extraction)
    Extracts text regions (Region of Interest - ROI) from the bottle image.
    This handles uneven lighting better than Part 1.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return [], None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive Thresholding handles uneven lighting on the cylindrical bottle
    thresh = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        15, 
        5
    )

    # Morphological closing connects characters into solid blocks
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rect_kernel)

    # Find contours
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois = []
    output_image = image.copy()

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = w * h

        # Filter out noise (boxes that are too small or not shaped like text)
        if area > 300 and area < 50000 and aspect_ratio > 1.2:
            rois.append((x, y, w, h))
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite(output_path, output_image)
    print(f"Localized {len(rois)} potential text regions. Saved to {output_path}")
    
    return rois, gray

if __name__ == "__main__":
    localize_text("data/bottle1.png", "outputs/part2_localization.jpg")
