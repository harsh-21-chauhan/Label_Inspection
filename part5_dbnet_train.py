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
        # Loss 1: Probability map should match ground truth text regions
        loss_prob = self.bce(prob_map, gt_prob)

        # Loss 2: Binary map (after differentiable binarization) should be crisp
        loss_binary = self.bce(binary_map, gt_prob)

        # Loss 3: Threshold map accuracy (only where text exists)
        if gt_thresh_mask.sum() > 0:
            loss_thresh = self.l1(
                thresh_map * gt_thresh_mask,
                gt_thresh * gt_thresh_mask
            )
        else:
            loss_thresh = torch.tensor(0.0)

        total = loss_prob + self.alpha * loss_binary + self.beta * loss_thresh
        return total, loss_prob.item(), loss_binary.item(), loss_thresh.item()


def train_dbnet():
    """
    Phase 5: DBNet Training Script
    Trains the lightweight DBNet on our full augmented CTW1500 dataset.
    """
    print("=" * 60)
    print("Phase 5: DBNet (Fast Text Detection) Training")
    print("=" * 60)

    # Dataset paths
    ann_path = "data/CTW1500/annotations/train_ctw1500_maxlen100_v2.json"
    img_dir = "data/CTW1500/ctwtrain_text_image"
    aug_dir = "outputs/ctw1500_augmented/train"

    # Load dataset (using target_size=640 for balanced speed/accuracy)
    dataset = DBNetDataset(ann_path, img_dir, aug_dir, target_size=640)
    print(f"Loaded {len(dataset)} training images")

    # DataLoader with batch_size=2 (safe for CPU/limited GPU memory)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    # Initialize model, loss, and optimizer
    model = DBNet()
    criterion = DBNetLoss(alpha=1.0, beta=10.0)

    # Adam optimizer (converges faster than SGD for this architecture)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training config
    num_epochs = 3
    model.train()

    print(f"\nStarting Training ({num_epochs} epochs, {len(data_loader)} batches/epoch)...\n")

    for epoch in range(num_epochs):
        running_loss = 0.0
        batch_count = 0

        for images, gt_prob, gt_thresh, gt_thresh_mask in data_loader:
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

            if batch_count % 25 == 0:
                avg = running_loss / batch_count
                print(f"  Epoch [{epoch+1}/{num_epochs}] Batch [{batch_count}/{len(data_loader)}] "
                      f"Loss: {avg:.4f} (Prob: {l_prob:.4f}, Binary: {l_binary:.4f}, Thresh: {l_thresh:.4f})")

        epoch_loss = running_loss / batch_count
        print(f"\n  >>> Epoch [{epoch+1}/{num_epochs}] Average Loss: {epoch_loss:.4f}\n")

    # Save the trained model
    torch.save(model.state_dict(), "outputs/dbnet_trained.pth")
    print("=" * 60)
    print(f"Training Complete! Model saved to outputs/dbnet_trained.pth")
    print("=" * 60)


if __name__ == "__main__":
    train_dbnet()
