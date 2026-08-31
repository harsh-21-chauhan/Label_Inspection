import torch
import torch.nn as nn
import torchvision.models as models

def get_shufflenet_model(num_classes=2):
    """
    Returns the ShuffleNet V2 architecture as specified in the paper.
    ShuffleNet uses Depthwise Separable Convolutions to remain extremely
    fast and efficient for industrial real-time applications.
    """
    # Load the industrial standard ShuffleNet V2
    # We use pretrained=False here because we are building it from scratch,
    # but in a real industrial scenario with 10k images, pretrained=True is better.
    model = models.shufflenet_v2_x1_0(weights=None)
    
    # The default ShuffleNet outputs 1000 classes (for ImageNet).
    # We only have 2 classes (Defect-Free vs Defective).
    # So we replace the final Fully Connected (Linear) layer.
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

if __name__ == "__main__":
    # Quick test to ensure the model compiles
    model = get_shufflenet_model()
    
    # Create a dummy tensor representing a single 224x224 RGB image
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Run a Forward Pass
    output = model(dummy_input)
    print(f"Model compiled successfully!")
    print(f"Output shape: {output.shape} (Expected: [1, 2] for 2 classes)")
