import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import LabelDefectDataset, get_transforms
from model import get_shufflenet_model

def train_model():
    """
    Part 3: Deep Learning Training Script
    Trains the ShuffleNet V2 on our synthesized dataset.
    """
    print("--- Initializing Label Inspection Deep Learning Pipeline ---")
    
    # 1. Load the Dataset
    dataset = LabelDefectDataset(".", transform=get_transforms)
    
    if len(dataset) == 0:
        print("No images found. Cannot train.")
        return
        
    # Batch size is small because we only have a tiny synthesized dataset right now.
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 2. Initialize Model, Loss Function, and Optimizer
    model = get_shufflenet_model(num_classes=2)
    criterion = nn.CrossEntropyLoss()
    
    # As per the paper, using Stochastic Gradient Descent (SGD)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.00005)
    
    # 3. Training Loop (We will just run 5 epochs as a proof-of-concept)
    num_epochs = 5
    
    model.train() # Set model to training mode
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in dataloader:
            # Forward Pass: push images through the network
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward Pass: calculate gradients and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    print("--- Training Complete! Model is ready for industrial deployment. ---")
    # Normally we would save the model here: torch.save(model.state_dict(), "shufflenet_defect.pth")

if __name__ == "__main__":
    train_model()
