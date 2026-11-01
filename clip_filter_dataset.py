"""
CLIP-based Dataset Filter for Total-Text (and CTW1500)
Scores each image on how similar it is to "product label / packaging / bottle / can"
vs "storefront / building facade / outdoor billboard / street sign"
Keeps only product- and packaging-relevant images for label inspection training.

Usage:
    conda run -n opencv python clip_filter_dataset.py
    # or with custom threshold:
    conda run -n opencv python clip_filter_dataset.py --threshold 0.020
"""

import os
import re
import sys
import json
import shutil
import csv
import argparse
from PIL import Image
import torch
import clip


# ─── DEFAULT CONFIG FOR TOTAL-TEXT ───
DATASET_ROOT = "data/Total-Text"
TRAIN_IMG_DIR = os.path.join(DATASET_ROOT, "Train")
TEST_IMG_DIR = os.path.join(DATASET_ROOT, "Test")
ANN_DIR = os.path.join(DATASET_ROOT, "Annotation")

OUTPUT_DIR = "data/Total-Text_filtered"
DEFAULT_THRESHOLD = 0.020  # Score = (pos_sim - neg_sim). > 0.02 keeps top ~17-18% product images

# ─── PROMPTS OPTIMIZED FOR TOTAL-TEXT ───
POSITIVE_PROMPTS = [
    "a photo of a product label on a bottle or can",
    "a photo of text printed on packaging, a box, jar, or container",
    "a close-up photo of a beverage bottle, wine label, or soda can",
    "text on curved cylindrical packaging or food product",
    "printed branding and label text on a consumer packaged good",
    "a photo of a cup, carton, or bottle with product text",
]

NEGATIVE_PROMPTS = [
    "a photo of a storefront, shop entrance, building facade, or street view",
    "a photo of an outdoor billboard, street sign, or road sign",
    "a photo of a large banner or poster hanging on a building wall",
    "a photo of an indoor room, office, or restaurant dining area",
    "a photo of people, pedestrians, clothing, or vehicles on a street",
    "a photo of a document, book page, or certificate",
]


def parse_totaltext_polygons(filepath):
    """
    Parses a Total-Text polygonal annotation file (.txt).
    Returns list of polygons with format:
    [{'pts': [x1, y1, x2, y2, ...], 'bbox': [x, y, w, h], 'trans': str}, ...]
    """
    if not os.path.exists(filepath):
        return []

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    pattern = re.compile(
        r"x:\s*\[\[(.*?)\]\],\s*y:\s*\[\[(.*?)\]\].*?(?:transcriptions:\s*\[(.*?)\])?",
        re.DOTALL,
    )
    polygons = []
    for m in pattern.finditer(content):
        xs_str, ys_str = m.group(1), m.group(2)
        xs = [
            float(v)
            for v in xs_str.replace("\n", " ").split()
            if v.replace(".", "", 1).isdigit()
        ]
        ys = [
            float(v)
            for v in ys_str.replace("\n", " ").split()
            if v.replace(".", "", 1).isdigit()
        ]
        trans = m.group(3).strip() if m.group(3) else ""

        if len(xs) == len(ys) and len(xs) >= 3:
            pts = []
            for x, y in zip(xs, ys):
                pts.extend([round(x, 2), round(y, 2)])
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            bbox = [
                round(min_x, 2),
                round(min_y, 2),
                round(max_x - min_x, 2),
                round(max_y - min_y, 2),
            ]
            area = round((max_x - min_x) * (max_y - min_y), 2)
            polygons.append({
                "pts": pts,
                "bbox": bbox,
                "area": area,
                "trans": trans,
            })
    return polygons


def score_images(img_dir, split_name, model, preprocess, device, threshold):
    """Score every image in an image directory using CLIP."""
    print(f"\n{'='*60}")
    print(f"Scoring {split_name} images from: {img_dir}")
    print(f"{'='*60}")

    if not os.path.exists(img_dir):
        print(f"Directory not found: {img_dir}")
        return []

    # Tokenize prompts
    pos_tokens = clip.tokenize(POSITIVE_PROMPTS).to(device)
    neg_tokens = clip.tokenize(NEGATIVE_PROMPTS).to(device)

    with torch.no_grad():
        pos_features = model.encode_text(pos_tokens)
        neg_features = model.encode_text(neg_tokens)
        pos_features /= pos_features.norm(dim=-1, keepdim=True)
        neg_features /= neg_features.norm(dim=-1, keepdim=True)

    image_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    total_imgs = len(image_files)
    print(f"Found {total_imgs} images in {split_name}.")

    results = []
    for i, fname in enumerate(image_files):
        img_path = os.path.join(img_dir, fname)
        try:
            image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                img_features = model.encode_image(image)
                img_features /= img_features.norm(dim=-1, keepdim=True)

                pos_sim = (img_features @ pos_features.T).mean().item()
                neg_sim = (img_features @ neg_features.T).mean().item()
                score = pos_sim - neg_sim

            results.append({
                "split": split_name.lower(),
                "file": fname,
                "score": round(score, 4),
                "pos_sim": round(pos_sim, 4),
                "neg_sim": round(neg_sim, 4),
                "keep": score > threshold,
            })
        except Exception as e:
            print(f"  Error on {fname}: {e}")
            continue

        if (i + 1) % 100 == 0 or (i + 1) == total_imgs:
            kept_so_far = sum(1 for r in results if r["keep"])
            print(f"  Processed {i+1}/{total_imgs} | Kept: {kept_so_far} ({kept_so_far/(i+1)*100:.1f}%)")

    return results


def filter_and_copy_totaltext(results, src_img_dir, split_name, output_dir):
    """
    Copies kept images and corresponding annotations.
    Also produces standard COCO format JSON for seamless merging with CTW1500.
    """
    split_lower = split_name.lower()
    out_img_dir = os.path.join(output_dir, f"{split_lower}_images")
    os.makedirs(out_img_dir, exist_ok=True)

    # Subfolders for raw annotations
    poly_ann_src = os.path.join(ANN_DIR, "groundtruth_polygonal_annotation", split_name)
    poly_ann_dst = os.path.join(output_dir, "Annotation", "groundtruth_polygonal_annotation", split_name)
    os.makedirs(poly_ann_dst, exist_ok=True)

    textregion_src = os.path.join(ANN_DIR, "groundtruth_textregion", split_name)
    textregion_dst = os.path.join(output_dir, "Annotation", "groundtruth_textregion", split_name)
    has_textregion = os.path.exists(textregion_src)
    if has_textregion:
        os.makedirs(textregion_dst, exist_ok=True)

    kept = [r for r in results if r["keep"]]

    coco_images = []
    coco_annotations = []
    ann_id_counter = 1

    for img_idx, r in enumerate(kept, start=1):
        fname = r["file"]
        stem, ext = os.path.splitext(fname)

        # 1. Copy image
        src_path = os.path.join(src_img_dir, fname)
        dst_path = os.path.join(out_img_dir, fname)
        shutil.copy2(src_path, dst_path)

        # 2. Copy polygon txt
        poly_file = f"poly_gt_{stem}.txt"
        poly_src_path = os.path.join(poly_ann_src, poly_file)
        if os.path.exists(poly_src_path):
            shutil.copy2(poly_src_path, os.path.join(poly_ann_dst, poly_file))

        # 3. Copy textregion mask if present
        if has_textregion:
            mask_file = f"{stem}.png"
            mask_src_path = os.path.join(textregion_src, mask_file)
            if os.path.exists(mask_src_path):
                shutil.copy2(mask_src_path, os.path.join(textregion_dst, mask_file))

        # 4. Read image dimensions for COCO metadata
        try:
            with Image.open(src_path) as im:
                width, height = im.size
        except Exception:
            width, height = 0, 0

        coco_images.append({
            "id": img_idx,
            "file_name": fname,
            "width": width,
            "height": height,
            "score": r["score"],
            "dataset_origin": "Total-Text",
        })

        # 5. Parse polygons into COCO annotations
        polys = parse_totaltext_polygons(poly_src_path)
        for poly in polys:
            coco_annotations.append({
                "id": ann_id_counter,
                "image_id": img_idx,
                "category_id": 1,
                "iscrowd": 0,
                "area": poly["area"],
                "bbox": poly["bbox"],
                "segmentation": [poly["pts"]],
                "transcription": poly["trans"],
            })
            ann_id_counter += 1

    # 6. Save standard COCO-format JSON
    coco_output = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [{"id": 1, "name": "text"}],
    }

    ann_json_dir = os.path.join(output_dir, "annotations")
    os.makedirs(ann_json_dir, exist_ok=True)
    json_path = os.path.join(ann_json_dir, f"{split_lower}_filtered.json")
    with open(json_path, "w") as f:
        json.dump(coco_output, f, indent=2)

    print(f"\n  [{split_name}] Kept: {len(kept)}/{len(results)} images")
    print(f"  [{split_name}] Kept text polygon annotations: {len(coco_annotations)}")
    print(f"  [{split_name}] Saved images to: {out_img_dir}")
    print(f"  [{split_name}] Saved COCO JSON to: {json_path}")

    return kept


def save_scores_csv(all_results, output_dir):
    """Save all image CLIP scores to CSV for manual inspection."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "all_scores.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "file", "score", "pos_sim", "neg_sim", "keep"])
        for r in sorted(all_results, key=lambda x: x["score"], reverse=True):
            writer.writerow([r["split"], r["file"], r["score"], r["pos_sim"], r["neg_sim"], r["keep"]])
    print(f"\nFull scores saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="CLIP-Based Dataset Filter for Total-Text")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Score threshold (pos_sim - neg_sim), default={DEFAULT_THRESHOLD}")
    parser.add_argument("--data_dir", type=str, default=DATASET_ROOT,
                        help="Path to Total-Text dataset root")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="Path to output directory for filtered dataset")
    parser.add_argument("--force_rescore", action="store_true",
                        help="Ignore existing scores CSV and re-score images with CLIP")
    args = parser.parse_args()

    threshold = args.threshold
    output_dir = args.output_dir
    train_dir = os.path.join(args.data_dir, "Train")
    test_dir = os.path.join(args.data_dir, "Test")

    print("=" * 60)
    print("CLIP-Based Dataset Filter for Total-Text")
    print(f"Target Data Root:  {args.data_dir}")
    print(f"Output Directory:  {output_dir}")
    print(f"Threshold:         {threshold} (pos_sim - neg_sim)")
    print("=" * 60)

    # Check if precomputed scores exist
    csv_path = os.path.join(output_dir, "all_scores.csv")
    if os.path.exists(csv_path) and not args.force_rescore:
        print(f"\nFound existing {csv_path} - loading precomputed scores...")
        train_results, test_results = [], []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                item = {
                    "split": row["split"],
                    "file": row["file"],
                    "score": float(row["score"]),
                    "pos_sim": float(row["pos_sim"]),
                    "neg_sim": float(row["neg_sim"]),
                    "keep": float(row["score"]) > threshold,
                }
                if row["split"] == "train":
                    train_results.append(item)
                else:
                    test_results.append(item)
    else:
        # Load CLIP
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"\nLoading CLIP model (ViT-B/32) on: {device} ...")
        model, preprocess = clip.load("ViT-B/32", device=device)

        train_results = score_images(train_dir, "Train", model, preprocess, device, threshold)
        test_results = score_images(test_dir, "Test", model, preprocess, device, threshold)

    all_results = train_results + test_results

    # Filter and copy images and annotations
    train_kept = filter_and_copy_totaltext(train_results, train_dir, "Train", output_dir)
    test_kept = filter_and_copy_totaltext(test_results, test_dir, "Test", output_dir)

    # Save / update CSV
    save_scores_csv(all_results, output_dir)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Train: {len(train_kept)}/{len(train_results)} images kept")
    print(f"  Test:  {len(test_kept)}/{len(test_results)} images kept")
    print(f"  Total: {len(train_kept) + len(test_kept)} filtered images")
    print(f"\n  Output Directory: {output_dir}/")
    print(f"  Scores CSV:       {output_dir}/all_scores.csv")
    print(f"\n  Top 5 BEST (most product/label-like):")
    sorted_res = sorted(all_results, key=lambda x: x["score"], reverse=True)
    for r in sorted_res[:5]:
        print(f"    [{r['split']}] {r['file']}  score={r['score']}")
    print(f"\n  Top 5 WORST (most building/sign-like):")
    for r in sorted_res[-5:]:
        print(f"    [{r['split']}] {r['file']}  score={r['score']}")
    print(f"{'='*60}")
    print("\nNext step: Review data/Total-Text_filtered/all_scores.csv, then run dataset merge.")


if __name__ == "__main__":
    main()
