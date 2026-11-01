import torch
import torch.nn as nn
import torchvision.models as models

def get_shufflenet_model(num_classes=2, pretrained=True):
    """
    Returns the ShuffleNet V2 architecture as specified in the paper.
    ShuffleNet uses Depthwise Separable Convolutions and Channel Shuffle
    to remain extremely fast (< 5ms) and efficient for industrial real-time applications.
    """
    weights = models.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
    model = models.shufflenet_v2_x1_0(weights=weights)

    # Replace final linear classifier layer
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
