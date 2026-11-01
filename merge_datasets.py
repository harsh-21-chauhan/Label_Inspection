"""
Merge and Synchronize Filtered Datasets:
1. CTW1500_filtered
2. Total-Text_filtered
3. scraped_labels_coco (New auto-annotated industrial bottle images)

Combines into a single unified COCO-compliant dataset: data/curated_dataset/
"""

import os
import json
import shutil
from PIL import Image

CTW_DIR = "data/CTW1500_filtered"
TT_DIR = "data/Total-Text_filtered"
SCRAPED_DIR = "data/scraped_labels_coco"
OUTPUT_DIR = "data/curated_dataset"


def clean_and_sync_split(split_name):
    """
    Reads disk images for CTW1500, Total-Text, and scraped Google labels,
    matches with annotations, re-indexes IDs, and outputs merged dataset.
    """
    out_img_dir = os.path.join(OUTPUT_DIR, f"{split_name}_images")
    os.makedirs(out_img_dir, exist_ok=True)

    merged_images = []
    merged_annotations = []
    img_id_counter = 1
    ann_id_counter = 1

    stats = {"ctw_kept": 0, "tt_kept": 0, "scraped_kept": 0}

    # 1. Process CTW1500
    ctw_img_folder = os.path.join(CTW_DIR, f"{split_name}_images")
    ctw_json_path = os.path.join(CTW_DIR, "annotations", f"{split_name}_filtered.json")

    if os.path.exists(ctw_img_folder) and os.path.exists(ctw_json_path):
        with open(ctw_json_path, "r") as f:
            ctw_data = json.load(f)

        disk_files = set(
            f for f in os.listdir(ctw_img_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png")) and not f.startswith(".")
        )

        ctw_ann_map = {}
        for ann in ctw_data.get("annotations", []):
            ctw_ann_map.setdefault(ann["image_id"], []).append(ann)

        for img_entry in ctw_data.get("images", []):
            fname = img_entry["file_name"]
            if fname not in disk_files:
                continue

            src_path = os.path.join(ctw_img_folder, fname)
            new_fname = f"ctw_{fname}"
            dst_path = os.path.join(out_img_dir, new_fname)
            shutil.copy2(src_path, dst_path)

            old_id = img_entry["id"]
            merged_images.append({
                "id": img_id_counter,
                "file_name": new_fname,
                "width": img_entry.get("width", 0),
                "height": img_entry.get("height", 0),
                "source": "CTW1500",
            })

            for ann in ctw_ann_map.get(old_id, []):
                new_ann = dict(ann)
                new_ann["id"] = ann_id_counter
                new_ann["image_id"] = img_id_counter
                merged_annotations.append(new_ann)
                ann_id_counter += 1

            img_id_counter += 1
            stats["ctw_kept"] += 1

    # 2. Process Total-Text
    tt_img_folder = os.path.join(TT_DIR, f"{split_name}_images")
    tt_json_path = os.path.join(TT_DIR, "annotations", f"{split_name}_filtered.json")

    if os.path.exists(tt_img_folder) and os.path.exists(tt_json_path):
        with open(tt_json_path, "r") as f:
            tt_data = json.load(f)

        disk_files = set(
            f for f in os.listdir(tt_img_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png")) and not f.startswith(".")
        )

        tt_ann_map = {}
        for ann in tt_data.get("annotations", []):
            tt_ann_map.setdefault(ann["image_id"], []).append(ann)

        for img_entry in tt_data.get("images", []):
            fname = img_entry["file_name"]
            if fname not in disk_files:
                continue

            src_path = os.path.join(tt_img_folder, fname)
            new_fname = f"tt_{fname}"
            dst_path = os.path.join(out_img_dir, new_fname)
            shutil.copy2(src_path, dst_path)

            old_id = img_entry["id"]
            merged_images.append({
                "id": img_id_counter,
                "file_name": new_fname,
                "width": img_entry.get("width", 0),
                "height": img_entry.get("height", 0),
                "source": "Total-Text",
            })

            for ann in tt_ann_map.get(old_id, []):
                new_ann = dict(ann)
                new_ann["id"] = ann_id_counter
                new_ann["image_id"] = img_id_counter
                merged_annotations.append(new_ann)
                ann_id_counter += 1

            img_id_counter += 1
            stats["tt_kept"] += 1

    # 3. Process Scraped Labels (Add to train split)
    if split_name == "train" and os.path.exists(SCRAPED_DIR):
        scraped_img_folder = os.path.join(SCRAPED_DIR, "train_images")
        scraped_json_path = os.path.join(SCRAPED_DIR, "annotations", "train_google_labels.json")

        if os.path.exists(scraped_img_folder) and os.path.exists(scraped_json_path):
            with open(scraped_json_path, "r") as f:
                scraped_data = json.load(f)

            disk_files = set(
                f for f in os.listdir(scraped_img_folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png")) and not f.startswith(".")
            )

            scraped_ann_map = {}
            for ann in scraped_data.get("annotations", []):
                scraped_ann_map.setdefault(ann["image_id"], []).append(ann)

            for img_entry in scraped_data.get("images", []):
                fname = img_entry["file_name"]
                if fname not in disk_files:
                    continue

                src_path = os.path.join(scraped_img_folder, fname)
                new_fname = f"scraped_{fname}"
                dst_path = os.path.join(out_img_dir, new_fname)
                shutil.copy2(src_path, dst_path)

                old_id = img_entry["id"]
                merged_images.append({
                    "id": img_id_counter,
                    "file_name": new_fname,
                    "width": img_entry.get("width", 0),
                    "height": img_entry.get("height", 0),
                    "source": "WebScraped_Industrial",
                })

                for ann in scraped_ann_map.get(old_id, []):
                    new_ann = dict(ann)
                    new_ann["id"] = ann_id_counter
                    new_ann["image_id"] = img_id_counter
                    merged_annotations.append(new_ann)
                    ann_id_counter += 1

                img_id_counter += 1
                stats["scraped_kept"] += 1

    # Save merged JSON
    out_ann_dir = os.path.join(OUTPUT_DIR, "annotations")
    os.makedirs(out_ann_dir, exist_ok=True)
    out_json_path = os.path.join(out_ann_dir, f"{split_name}_merged.json")

    merged_data = {
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": [{"id": 1, "name": "text"}],
    }

    with open(out_json_path, "w") as f:
        json.dump(merged_data, f, indent=2)

    print(f"\n--- [{split_name.upper()}] MERGE SUMMARY ---")
    print(f"  CTW1500 images:     {stats['ctw_kept']}")
    print(f"  Total-Text images:  {stats['tt_kept']}")
    if split_name == "train":
        print(f"  Scraped Bottle images: {stats['scraped_kept']}")
    print(f"  Total {split_name} images: {len(merged_images)}")
    print(f"  Total annotations:  {len(merged_annotations)}")
    print(f"  Saved JSON to:      {out_json_path}")

    return len(merged_images), len(merged_annotations)


def main():
    print("=" * 60)
    print("MERGING & SYNCHRONIZING ALL CURATED DATASETS")
    print("=" * 60)

    train_imgs, train_anns = clean_and_sync_split("train")
    test_imgs, test_anns = clean_and_sync_split("test")

    print("\n" + "=" * 60)
    print("FINAL UNIFIED DATASET STATS (data/curated_dataset/)")
    print("=" * 60)
    print(f"  Train Images:       {train_imgs}")
    print(f"  Train Annotations:  {train_anns}")
    print(f"  Test Images:        {test_imgs}")
    print(f"  Test Annotations:   {test_anns}")
    print(f"  Total Dataset Size: {train_imgs + test_imgs} images")
    print(f"  Total Annotations:  {train_anns + test_anns} text regions")
    print("=" * 60)


if __name__ == "__main__":
    main()
