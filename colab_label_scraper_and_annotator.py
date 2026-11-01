"""
=============================================================================
Google Colab Industrial Bottle & Label Scraper + Auto-Annotator (EasyOCR)
=============================================================================
Run this entire script inside a Google Colab notebook (T4 GPU).
1. Searches and downloads real photos of industrial product labels.
2. Runs EasyOCR on GPU to auto-detect text polygon regions.
3. Packages everything into `scraped_labels_coco.zip` with standard COCO annotations.

Setup in Colab:
    !pip install -q easyocr icrawler pillow shapely matplotlib opencv-python
=============================================================================
"""

import os
import json
import shutil
import logging
from PIL import Image

SEARCH_QUERIES = [
    "medicine bottle label close up photo real",
    "prescription pill bottle label printed text photo",
    "pharmaceutical cough syrup amber bottle label inspection",
    "beer bottle label macro close up photo",
    "wine bottle label front view photography real",
    "aluminum soda can label curved packaging printed text photo",
    "shampoo bottle label ingredients printed text photo",
    "cosmetic cream jar label packaging photo close up",
    "canned food label tin can packaging photo",
    "energy drink can nutrition label close up photo",
]

IMAGES_PER_QUERY = 25
RAW_DIR = "scraped_raw"
FINAL_IMG_DIR = "google_labels/train_images"
FINAL_ANN_DIR = "google_labels/annotations"


def step1_download_images():
    print("=" * 60)
    print("STEP 1: Downloading images from search engine...")
    print("=" * 60)
    from icrawler.builtin import BingImageCrawler

    # Suppress crawler network warning logs (403/502 from third-party sites)
    logging.getLogger("downloader").setLevel(logging.CRITICAL)
    logging.getLogger("parser").setLevel(logging.CRITICAL)
    logging.getLogger("crawler").setLevel(logging.CRITICAL)

    os.makedirs(RAW_DIR, exist_ok=True)

    for i, query in enumerate(SEARCH_QUERIES, 1):
        print(f"[{i}/{len(SEARCH_QUERIES)}] Downloading: '{query}'")
        temp_folder = os.path.join(RAW_DIR, f"query_{i:02d}")
        os.makedirs(temp_folder, exist_ok=True)

        crawler = BingImageCrawler(storage={"root_dir": temp_folder}, log_level=logging.CRITICAL)
        crawler.crawl(keyword=query, max_num=IMAGES_PER_QUERY, min_size=(400, 400))

    total_downloaded = sum(len(files) for _, _, files in os.walk(RAW_DIR))
    print(f"Download complete! Successfully collected {total_downloaded} candidate images.")


def step2_auto_annotate_with_ocr():
    print("\n" + "=" * 60)
    print("STEP 2: Auto-Annotating images using EasyOCR...")
    print("=" * 60)
    import easyocr
    import torch

    use_gpu = torch.cuda.is_available()
    print(f"Initializing EasyOCR (GPU: {use_gpu})...")
    reader = easyocr.Reader(["en"], gpu=use_gpu)

    os.makedirs(FINAL_IMG_DIR, exist_ok=True)
    os.makedirs(FINAL_ANN_DIR, exist_ok=True)

    coco_images = []
    coco_annotations = []
    img_id = 1
    ann_id = 1
    rejected_no_text = 0

    all_files = []
    for root, _, files in os.walk(RAW_DIR):
        for f in sorted(files):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                all_files.append(os.path.join(root, f))

    print(f"Scanning and annotating {len(all_files)} images on GPU...")

    for idx, src_path in enumerate(all_files, 1):
        try:
            with Image.open(src_path) as im:
                im_rgb = im.convert("RGB")
                w, h = im_rgb.size
                if w / h > 3.8 or h / w > 3.8:
                    continue
        except Exception:
            continue

        try:
            results = reader.readtext(src_path)
        except Exception as e:
            print(f"  [Image {idx}] Error: {e}")
            continue

        valid_detections = [item for item in results if item[2] >= 0.15]
        if len(valid_detections) < 2:
            rejected_no_text += 1
            continue

        new_fname = f"bottle_label_{img_id:04d}.jpg"
        dst_path = os.path.join(FINAL_IMG_DIR, new_fname)
        im_rgb.save(dst_path, "JPEG", quality=95)

        coco_images.append({
            "id": img_id,
            "file_name": new_fname,
            "width": w,
            "height": h,
            "source": "WebScraped_Industrial"
        })

        for box, text_val, conf in valid_detections:
            pts = []
            for pt in box:
                pts.extend([round(float(pt[0]), 2), round(float(pt[1]), 2)])

            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            bw = round(float(max_x - min_x), 2)
            bh = round(float(max_y - min_y), 2)
            area = round(bw * bh, 2)

            coco_annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "iscrowd": 0,
                "area": area,
                "bbox": [round(float(min_x), 2), round(float(min_y), 2), bw, bh],
                "segmentation": [pts],
                "transcription": str(text_val),
                "confidence": round(float(conf), 3)
            })
            ann_id += 1

        img_id += 1
        if img_id % 10 == 0:
            print(f"  ✓ Annotated {img_id-1} bottle labels ({ann_id-1} text regions detected)...")

    out_json = os.path.join(FINAL_ANN_DIR, "train_google_labels.json")
    with open(out_json, "w") as fp:
        json.dump({
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": [{"id": 1, "name": "text"}]
        }, fp, indent=2)

    print("\n" + "=" * 60)
    print("AUTO-ANNOTATION COMPLETE SUMMARY:")
    print(f"  Valid Annotated Bottle Labels: {len(coco_images)}")
    print(f"  Total Text Polygon Annotations: {len(coco_annotations)}")
    print(f"  Rejected (Too few text regions): {rejected_no_text}")
    print(f"  Saved COCO Annotations to:      {out_json}")
    print("=" * 60)

    # Package into zip
    zip_filename = "scraped_labels_coco"
    shutil.make_archive(zip_filename, "zip", "google_labels")
    file_size_mb = os.path.getsize(f"{zip_filename}.zip") / (1024 * 1024)
    print(f"\n✓ Package created: {zip_filename}.zip ({file_size_mb:.2f} MB)")


if __name__ == "__main__":
    step1_download_images()
    step2_auto_annotate_with_ocr()
