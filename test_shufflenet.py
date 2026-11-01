import os
import time
import cv2
import torch
import numpy as np
from dataset import get_defect_dataloader
from model import get_shufflenet_model

def evaluate_shufflenet(model_path="outputs/shufflenet_trained.pth", data_dir="data/defect_dataset", output_dir="outputs/shufflenet_test_results"):
    """
    Research-Grade Quantitative Evaluation for ShuffleNet V2:
      1. Multi-metric performance (Accuracy, Precision, Recall, Specificity, F1, F2).
      2. Defect sub-type breakdown (Missing Code vs Blurred Code vs Code Adhesion).
      3. Hardware latency benchmark (Inference time in ms, Throughput in FPS).
      4. 2x2 Industrial Confusion Matrix.
      5. Qualitative Visual Inspection Overlays and Montage.
    """
    print("=" * 70)
    print("ShuffleNet V2 Defect Classification: Research Evaluation Suite")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Hardware Accelerator: {device}")

    val_loader, val_ds = get_defect_dataloader(data_dir=data_dir, split="val", batch_size=16, shuffle=False)

    if val_loader is None or not os.path.exists(model_path):
        print(f"Error: Model {model_path} or validation data not found.")
        return

    num_classes = len(val_ds.classes)
    model = get_shufflenet_model(num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []

    # Inference loop
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # 1. Core Metrics
    total = len(all_labels)
    acc = 100.0 * (all_preds == all_labels).sum() / total

    tp = int(np.logical_and(all_preds == 1, all_labels == 1).sum())
    tn = int(np.logical_and(all_preds == 0, all_labels == 0).sum())
    fp = int(np.logical_and(all_preds == 1, all_labels == 0).sum())
    fn = int(np.logical_and(all_preds == 0, all_labels == 1).sum())

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    specificity = tn / (tn + fp + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    # F2 score weighs recall higher than precision (crucial in zero-defect manufacturing)
    f2 = 5 * precision * recall / (4 * precision + recall + 1e-6)

    # 2. Defect Sub-Type Breakdown
    sub_stats = {"missing": {"total": 0, "detected": 0},
                 "blur":    {"total": 0, "detected": 0},
                 "adhesion":{"total": 0, "detected": 0}}

    for idx, (path, true_lbl) in enumerate(val_ds.samples):
        if true_lbl == 1:
            for k in sub_stats:
                if k in path:
                    sub_stats[k]["total"] += 1
                    if all_preds[idx] == 1:
                        sub_stats[k]["detected"] += 1

    # 3. Hardware Latency Benchmark
    dummy = torch.randn(1, 3, 224, 224).to(device)
    for _ in range(15): _ = model(dummy) # warmup
    if device.type == "mps": torch.mps.synchronize()
    
    t0 = time.time()
    n_runs = 100
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(dummy)
    if device.type == "mps": torch.mps.synchronize()
    lat_ms = (time.time() - t0) / n_runs * 1000.0
    fps = 1000.0 / lat_ms

    # 4. Display Publication-Ready Research Summary
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    print("\n" + "=" * 70)
    print("TABLE 1: PRIMARY CLASSIFICATION METRICS (Unseen Test Patches)")
    print("=" * 70)
    print(f"  Overall Accuracy:      {acc:.2f}%")
    print(f"  Defect Precision:      {precision * 100:.2f}% (False alarm rate: {(1 - precision)*100:.2f}%)")
    print(f"  Defect Recall:         {recall * 100:.2f}% (Defects successfully caught)")
    print(f"  Normal Specificity:    {specificity * 100:.2f}% (Good bottles approved)")
    print(f"  F1-Score:              {f1:.4f}")
    print(f"  F2-Score (Safety):     {f2:.4f} (Higher penalty on missed defects)")

    print("\n" + "=" * 70)
    print("TABLE 2: ABLATION ACROSS PRINT DEFECT SUB-TYPES (Section 3.3.3)")
    print("=" * 70)
    for k, d in sub_stats.items():
        sub_rate = d["detected"] / max(d["total"], 1) * 100.0
        label_map = {"blur": "Blurred Code (Defocus / Ink Spread)",
                     "missing": "Missing Code (Faint Print / Broken Digits)",
                     "adhesion": "Code Adhesion (Excess Ink / Dot Bridging)"}
        print(f"  {label_map[k]:42s}: {d['detected']:2d} / {d['total']:2d} caught ({sub_rate:5.1f}%)")

    print("\n" + "=" * 70)
    print("TABLE 3: EDGE INFERENCE BENCHMARK & HARDWARE FEASIBILITY")
    print("=" * 70)
    print(f"  Architecture:          ShuffleNet V2 (1.0x Depthwise Separable CNN)")
    print(f"  Model Parameters:      {total_params:,} (~1.26 Million)")
    print(f"  Model Storage Size:    {model_size_mb:.2f} MB")
    print(f"  Inference Latency:     {lat_ms:.2f} ms per bottle patch")
    print(f"  Maximum Throughput:    {fps:.1f} FPS (>{fps * 60:.0f} bottles/minute capacity)")

    print("\n" + "=" * 70)
    print("2x2 INDUSTRIAL CONFUSION MATRIX")
    print("=" * 70)
    print(f"  True Negatives (Normal Pass):      {tn:3d} / 60  ({tn/60*100:.1f}%)")
    print(f"  False Alarms (False Rejections):   {fp:3d} / 60  ({fp/60*100:.1f}%)")
    print(f"  True Positives (Defects Caught):   {tp:3d} / 180 ({tp/180*100:.1f}%)")
    print(f"  Missed Defects (False Negatives):  {fn:3d} / 180 ({fn/180*100:.1f}%)")
    print("=" * 70)

    # 5. Generate Qualitative Visual Overlays and 4x4 Inspection Grid
    print(f"\nGenerating visual inspection overlays in {output_dir}/...")
    sample_indices = np.random.choice(total, min(24, total), replace=False)
    grid_tiles = []

    for idx in sample_indices:
        img_path, true_lbl = val_ds.samples[idx]
        pred_lbl = all_preds[idx]
        conf = all_probs[idx][pred_lbl] * 100.0

        mat = cv2.imread(img_path)
        if mat is None:
            continue
        mat = cv2.resize(mat, (224, 224))

        color = (0, 200, 0) if pred_lbl == 0 else (0, 0, 220)
        label_text = f"PASS: {conf:.1f}%" if pred_lbl == 0 else f"DEFECT: {conf:.1f}%"

        cv2.rectangle(mat, (0, 0), (223, 223), color, 4)
        cv2.rectangle(mat, (0, 0), (224, 32), color, -1)
        cv2.putText(mat, label_text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        is_correct = (pred_lbl == true_lbl)
        status = "CORRECT" if is_correct else "MISMATCH"
        status_color = (0, 255, 0) if is_correct else (0, 0, 255)
        cv2.putText(mat, f"True: {'Norm' if true_lbl==0 else 'Def'} ({status})", (8, 212), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)

        fname = os.path.basename(img_path)
        cv2.imwrite(os.path.join(output_dir, f"inspect_{fname}"), mat)
        grid_tiles.append(mat)

    if len(grid_tiles) >= 16:
        rows = [np.hstack(grid_tiles[i:i+4]) for i in range(0, 16, 4)]
        montage = np.vstack(rows)
        cv2.imwrite(os.path.join(output_dir, "inspection_montage.jpg"), montage)
        print(f"  Montage ready for slides: {output_dir}/inspection_montage.jpg")

    print("=" * 70)

if __name__ == "__main__":
    evaluate_shufflenet()
