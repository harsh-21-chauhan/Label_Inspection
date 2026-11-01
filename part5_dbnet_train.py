import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from part5_dbnet_model import DBNet
from part5_dbnet_dataset import DBNetDataset


class DBNetLoss(nn.Module):
    """
    DBNet uses a combined loss of 3 components:
    1. Probability Loss (BCE): How well does the model predict text vs background?
    2. Threshold Loss (L1): How accurately does the model learn pixel-level thresholds?
    3. Binary Loss (BCE): How crisp is the final binarized output?

    Total Loss = L_prob + alpha * L_binary + beta * L_thresh
    """
    def __init__(self, alpha=1.0, beta=10.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce = nn.BCELoss(reduction='mean')
        self.l1 = nn.L1Loss(reduction='mean')

    def forward(self, prob_map, thresh_map, binary_map, gt_prob, gt_thresh, gt_thresh_mask):
        loss_prob = self.bce(prob_map, gt_prob)
        loss_binary = self.bce(binary_map, gt_prob)

        if gt_thresh_mask.sum() > 0:
            loss_thresh = self.l1(
                thresh_map * gt_thresh_mask,
                gt_thresh * gt_thresh_mask
            )
        else:
            loss_thresh = torch.tensor(0.0, device=prob_map.device)

        total = loss_prob + self.alpha * loss_binary + self.beta * loss_thresh
        return total, loss_prob.item(), loss_binary.item(), loss_thresh.item()


def train_dbnet(epochs=10, batch_size=4, lr=0.001, save_path="outputs/dbnet_trained.pth"):
    print("=" * 60)
    print("Phase 5: DBNet (Fast Text Detection) Training")
    print("=" * 60)

    # Device configuration (Auto-detects Apple Silicon MPS, CUDA GPU, or CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Hardware Accelerator: {device}")

    # Dataset paths (Curated bottle & packaging dataset)
    ann_path = "data/curated_dataset/annotations/train_bottles_curated.json"
    img_dir = "data/curated_dataset/train_images"

    os.makedirs("outputs", exist_ok=True)

    # Load dataset (with dynamic on-the-fly industrial augmentations)
    dataset = DBNetDataset(ann_path, img_dir, target_size=640)
    print(f"Loaded {len(dataset)} training samples ({len(dataset.images)} curated bottles x 3 dynamic variants)")

    # DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Initialize model with ImageNet pretraining, loss, and optimizer
    model = DBNet(pretrained=True).to(device)
    criterion = DBNetLoss(alpha=1.0, beta=10.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    num_epochs = epochs
    model.train()

    print(f"\nStarting Training ({num_epochs} epochs, {len(data_loader)} batches/epoch)...\n")

    for epoch in range(num_epochs):
        running_loss = 0.0
        batch_count = 0

        for images, gt_prob, gt_thresh, gt_thresh_mask in data_loader:
            images = images.to(device)
            gt_prob = gt_prob.to(device)
            gt_thresh = gt_thresh.to(device)
            gt_thresh_mask = gt_thresh_mask.to(device)

            # Forward pass
            prob_map, thresh_map, binary_map = model(images)

            # Ensure output matches ground truth size
            if prob_map.shape != gt_prob.shape:
                gt_prob = torch.nn.functional.interpolate(gt_prob, size=prob_map.shape[2:], mode='bilinear', align_corners=False)
                gt_thresh = torch.nn.functional.interpolate(gt_thresh, size=thresh_map.shape[2:], mode='bilinear', align_corners=False)
                gt_thresh_mask = torch.nn.functional.interpolate(gt_thresh_mask, size=thresh_map.shape[2:], mode='bilinear', align_corners=False)

            # Calculate loss
            total_loss, l_prob, l_binary, l_thresh = criterion(
                prob_map, thresh_map, binary_map, gt_prob, gt_thresh, gt_thresh_mask
            )

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            batch_count += 1

            if batch_count % 25 == 0 or batch_count == len(data_loader):
                avg = running_loss / batch_count
                print(f"  Epoch [{epoch+1}/{num_epochs}] Batch [{batch_count}/{len(data_loader)}] "
                      f"Loss: {avg:.4f} (Prob: {l_prob:.4f}, Binary: {l_binary:.4f}, Thresh: {l_thresh:.4f})")

        epoch_loss = running_loss / max(batch_count, 1)
        print(f"\n  >>> Epoch [{epoch+1}/{num_epochs}] Average Loss: {epoch_loss:.4f}\n")

    # Save the trained model
    torch.save(model.state_dict(), save_path)
    print("=" * 60)
    print(f"Training Complete! Model saved to {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DBNet on Curated Label Dataset")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--save_path", type=str, default="outputs/dbnet_trained.pth", help="Checkpoint save path")
    args = parser.parse_args()

    train_dbnet(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, save_path=args.save_path)
