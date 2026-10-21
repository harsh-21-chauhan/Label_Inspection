"""
CLIP-based Dataset Filter for CTW1500
Scores each image on how similar it is to "product label on a bottle"
vs "street sign or poster or document" — keeps only product-relevant images.

Usage:
    conda run -n opencv python clip_filter_dataset.py
"""

import os
import json
import shutil
import csv
import torch
import clip
from PIL import Image

# ─── CONFIG ───
TRAIN_IMG_DIR = "data/CTW1500/ctwtrain_text_image"
TEST_IMG_DIR = "data/CTW1500/ctwtest_text_image"
TRAIN_ANN = "data/CTW1500/annotations/train_ctw1500_maxlen100_v2.json"
TEST_ANN = "data/CTW1500/annotations/test_ctw1500_maxlen100.json"

OUTPUT_DIR = "data/CTW1500_filtered"
SCORE_THRESHOLD = 0.020  # Keep images scoring above 0.02 (pos_sim - neg_sim)

# Positive prompts (what we WANT)
POSITIVE_PROMPTS = [
    "a photo of a product label on a bottle",
    "a photo of text printed on a beverage container",
    "a close-up of a label on a wine bottle or beer can",
    "text on curved packaging or cylindrical product",
    "printed text on a food or medicine package",
]

# Negative prompts (what we DON'T want)
NEGATIVE_PROMPTS = [
    "a photo of a street sign or road sign",
    "a photo of a poster or banner on a wall",
    "a photo of a document or certificate",
    "a photo of a shop front or storefront sign",
    "a photo of a billboard or advertisement board",
]


def score_images(img_dir, split_name):
    """Score every image in a directory using CLIP."""
    print(f"\n{'='*60}")
    print(f"Scoring {split_name} images...")
    print(f"{'='*60}")

    # Load CLIP model (ViT-B/32 is fast and good enough for filtering)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    print(f"CLIP loaded on: {device}")

    # Tokenize all prompts
    pos_tokens = clip.tokenize(POSITIVE_PROMPTS).to(device)
    neg_tokens = clip.tokenize(NEGATIVE_PROMPTS).to(device)

    with torch.no_grad():
        pos_features = model.encode_text(pos_tokens)
        neg_features = model.encode_text(neg_tokens)
        pos_features /= pos_features.norm(dim=-1, keepdim=True)
        neg_features /= neg_features.norm(dim=-1, keepdim=True)

    # Score each image
    results = []
    image_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    
    for i, fname in enumerate(image_files):
        img_path = os.path.join(img_dir, fname)
        try:
            image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                img_features = model.encode_image(image)
                img_features /= img_features.norm(dim=-1, keepdim=True)

                # Cosine similarity with positive and negative prompts
                pos_sim = (img_features @ pos_features.T).mean().item()
                neg_sim = (img_features @ neg_features.T).mean().item()

                # Score = how much more "product-like" than "sign-like"
                score = pos_sim - neg_sim
                
            results.append({
                'file': fname,
                'score': round(score, 4),
                'pos_sim': round(pos_sim, 4),
                'neg_sim': round(neg_sim, 4),
                'keep': score > SCORE_THRESHOLD
            })
        except Exception as e:
            print(f"  Error on {fname}: {e}")
            continue

        if (i + 1) % 100 == 0:
            print(f"  Scored {i+1}/{len(image_files)} images...")

    print(f"  Done! Scored {len(results)} images.")
    return results


def filter_and_copy(results, img_dir, ann_path, split_name):
    """Copy kept images + filter annotations JSON to output directory."""
    out_img_dir = os.path.join(OUTPUT_DIR, f"{split_name}_images")
    os.makedirs(out_img_dir, exist_ok=True)

    kept = [r for r in results if r['keep']]
    kept_files = set(r['file'] for r in kept)

    # Copy images
    for r in kept:
        src = os.path.join(img_dir, r['file'])
        dst = os.path.join(out_img_dir, r['file'])
        shutil.copy2(src, dst)

    # Filter annotations JSON to only include kept images
    with open(ann_path, 'r') as f:
        coco = json.load(f)

    kept_images = [img for img in coco['images'] if img['file_name'] in kept_files]
    kept_img_ids = set(img['id'] for img in kept_images)
    kept_anns = [ann for ann in coco['annotations'] if ann['image_id'] in kept_img_ids]

    filtered_coco = {
        'images': kept_images,
        'annotations': kept_anns,
        'categories': coco.get('categories', [])
    }

    ann_out_dir = os.path.join(OUTPUT_DIR, "annotations")
    os.makedirs(ann_out_dir, exist_ok=True)
    ann_out_path = os.path.join(ann_out_dir, f"{split_name}_filtered.json")
    with open(ann_out_path, 'w') as f:
        json.dump(filtered_coco, f)

    print(f"\n  [{split_name}] Kept: {len(kept)}/{len(results)} images")
    print(f"  [{split_name}] Kept annotations: {len(kept_anns)}")
    print(f"  [{split_name}] Saved to: {out_img_dir}")

    return kept


def save_scores_csv(train_results, test_results):
    """Save all scores to CSV for manual review."""
    csv_path = os.path.join(OUTPUT_DIR, "all_scores.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['split', 'file', 'score', 'pos_sim', 'neg_sim', 'keep'])
        for r in sorted(train_results, key=lambda x: x['score'], reverse=True):
            writer.writerow(['train', r['file'], r['score'], r['pos_sim'], r['neg_sim'], r['keep']])
        for r in sorted(test_results, key=lambda x: x['score'], reverse=True):
            writer.writerow(['test', r['file'], r['score'], r['pos_sim'], r['neg_sim'], r['keep']])
    
    print(f"\nFull scores saved to: {csv_path}")


def main():
    print("=" * 60)
    print("CLIP-Based CTW1500 Dataset Filter")
    print(f"Threshold: {SCORE_THRESHOLD} (positive - negative similarity)")
    print("=" * 60)

    csv_path = os.path.join(OUTPUT_DIR, "all_scores.csv")
    if os.path.exists(csv_path):
        print(f"Found existing {csv_path} - loading precomputed scores...")
        train_results, test_results = [], []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                item = {
                    'file': row['file'],
                    'score': float(row['score']),
                    'pos_sim': float(row['pos_sim']),
                    'neg_sim': float(row['neg_sim']),
                    'keep': float(row['score']) > SCORE_THRESHOLD
                }
                if row['split'] == 'train':
                    train_results.append(item)
                else:
                    test_results.append(item)
    else:
        # Score both splits
        train_results = score_images(TRAIN_IMG_DIR, "TRAIN")
        test_results = score_images(TEST_IMG_DIR, "TEST")

    # Filter and copy
    train_kept = filter_and_copy(train_results, TRAIN_IMG_DIR, TRAIN_ANN, "train")
    test_kept = filter_and_copy(test_results, TEST_IMG_DIR, TEST_ANN, "test")

    # Save CSV for manual verification
    save_scores_csv(train_results, test_results)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Train: {len(train_kept)}/{len(train_results)} images kept")
    print(f"  Test:  {len(test_kept)}/{len(test_results)} images kept")
    print(f"  Total: {len(train_kept) + len(test_kept)} filtered images")
    print(f"\n  Output: {OUTPUT_DIR}/")
    print(f"  Scores CSV: {OUTPUT_DIR}/all_scores.csv")
    print(f"\n  Top 5 BEST (most product-like):")
    all_sorted = sorted(train_results + test_results, key=lambda x: x['score'], reverse=True)
    for r in all_sorted[:5]:
        print(f"    {r['file']}  score={r['score']}")
    print(f"\n  Top 5 WORST (most sign/poster-like):")
    for r in all_sorted[-5:]:
        print(f"    {r['file']}  score={r['score']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
