import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_defect_dataloader
from model import get_shufflenet_model

def train_shufflenet(epochs=10, batch_size=16, lr=0.01, save_path="outputs/shufflenet_trained.pth"):
    """
    Phase 4: ShuffleNet V2 Industrial Defect Classification Training
    Trains the lightweight ShuffleNet V2 model as specified in the reference paper:
      - Architecture: ShuffleNet V2 (1.0x) with Depthwise Separable Convolutions & Channel Shuffle
      - Optimizer: SGD with Momentum (0.9), weight decay (5e-5)
      - Loss: CrossEntropyLoss
    """
    print("=" * 60)
    print("Phase 4: ShuffleNet V2 Defect Classification Training")
    print("=" * 60)

    # Device configuration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Hardware Accelerator: {device}")

    # Load dataloaders
    train_loader, train_ds = get_defect_dataloader(data_dir="data/defect_dataset", split="train", batch_size=batch_size, shuffle=True)
    val_loader, val_ds = get_defect_dataloader(data_dir="data/defect_dataset", split="val", batch_size=batch_size, shuffle=False)

    if train_loader is None or len(train_ds) == 0:
        print("ERROR: Defect dataset not found in data/defect_dataset/.")
        print("Please run: conda run -n opencv python generate_defect_dataset.py first.")
        return

    print(f"Loaded {len(train_ds)} training patches across classes: {train_ds.class_to_idx}")
    print(f"Loaded {len(val_ds)} validation patches")

    # Initialize model with ImageNet pretraining
    num_classes = len(train_ds.classes)
    model = get_shufflenet_model(num_classes=num_classes, pretrained=True).to(device)

    # Compute inverse-frequency class weights to perfectly balance 240 normal vs 720 defective samples
    class_counts = [0] * num_classes
    for _, lbl in train_ds.samples:
        class_counts[lbl] += 1
    total_samples = len(train_ds)
    weights = [total_samples / (num_classes * max(c, 1)) for c in class_counts]
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"Inverse class weights: {class_weights.cpu().numpy()} (Normal={weights[0]:.2f}, Defective={weights[1]:.2f})")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer: As specified in Section 4 of the paper (SGD + Momentum + Weight Decay)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"\nStarting Training ({epochs} epochs, {len(train_loader)} batches/epoch, lr={lr})...\n")

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()
        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / max(val_total, 1)
        val_acc = 100.0 * val_correct / max(val_total, 1)

        print(f"Epoch [{epoch+1:2d}/{epochs:2d}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    print("\n" + "=" * 60)
    print(f"Training Complete! Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {save_path}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (default 0.01 as in paper)")
    args = parser.parse_args()

    train_shufflenet(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
