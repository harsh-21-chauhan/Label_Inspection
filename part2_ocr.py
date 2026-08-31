import cv2
import pytesseract
from part2_localization import localize_text

def run_ocr_on_regions(image_path, localization_output):
    """
    Part 2: Localized OCR
    Extracts text only from the specific regions identified in the localization step.
    """
    rois, gray = localize_text(image_path, localization_output)
    
    if gray is None or len(rois) == 0:
        print("No text regions found to process.")
        return

    print("\n--- OCR Results from Localized Regions ---")
    
    for i, (x, y, w, h) in enumerate(rois):
        # Crop the text region
        padding = 5
        y1 = max(0, y - padding)
        y2 = min(gray.shape[0], y + h + padding)
        x1 = max(0, x - padding)
        x2 = min(gray.shape[1], x + w + padding)
        
        roi_crop = gray[y1:y2, x1:x2]
        
        # Threshold the crop
        _, roi_thresh = cv2.threshold(roi_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Run Tesseract OCR (psm 7 treats image as a single text line)
        text = pytesseract.image_to_string(roi_thresh, config="--psm 7").strip()
        
        if text:
            print(f"Region {i+1} (x:{x}, y:{y}): {text}")
        else:
            print(f"Region {i+1} (x:{x}, y:{y}): [No text detected]")

    print("------------------------------------------")

if __name__ == "__main__":
    run_ocr_on_regions("data/bottle2.jpg", "outputs/part2_localization.jpg")
