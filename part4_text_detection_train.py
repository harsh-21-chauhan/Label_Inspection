import torch
from torch.utils.data import DataLoader
from part4_text_detection_dataset import CTW1500Dataset, collate_fn
from part4_text_detection_model import get_mask_rcnn_model

def train_text_detector():
    """
    Phase 4: Text Detection Training Loop
    Trains the Mask R-CNN on the CTW1500 curved text dataset and our
    synthetically degraded images to learn robust text detection.
    """
    print("--- Initializing Phase 4: Mask R-CNN Text Detection Training ---")
    
    img_dirs = {
        'original': "data/CTW1500/ctwtrain_text_image",
        'augmented': "outputs/ctw1500_augmented"
    }
    json_path = "data/CTW1500/annotations/train_ctw1500_maxlen100_v2.json"
    
    # 1. Load Dataset
    dataset = CTW1500Dataset(json_path, img_dirs)
    print(f"Loaded {len(dataset)} images (Original + Blur + Missing Variants)")
    
    # DataLoader handles batching
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    # 2. Initialize Model and Optimizer
    model = get_mask_rcnn_model(num_classes=2)
    model.train() # Mask R-CNN loss is automatically calculated during training mode
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    # 3. Training Loop (Proof of concept: 1 Epoch)
    print("\nStarting Training (1 Epoch)...")
    for images, targets in data_loader:
        # The model expects a list of images and a list of target dictionaries
        loss_dict = model(images, targets)
        
        # Mask R-CNN returns multiple losses (box loss, mask loss, etc.)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        print(f"Batch Loss: {losses.item():.4f} (Combines Bounding Box & Polygon Mask loss)")
        
    print("\n--- Training Complete! Model is learning curved text segmentation! ---")

if __name__ == "__main__":
    train_text_detector()
