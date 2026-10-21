import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class DBNetBackbone(nn.Module):
    """
    DBNet Backbone: Uses a lightweight ResNet-18 to extract multi-scale features.
    ResNet-18 is chosen over ResNet-50 because DBNet's entire purpose is SPEED.
    """
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None)

        # We split ResNet into 4 stages to get feature maps at different scales.
        # Each stage halves the resolution and doubles the channels.
        # Stage 1: 1/4 resolution  (64 channels)
        # Stage 2: 1/8 resolution  (128 channels)
        # Stage 3: 1/16 resolution (256 channels)
        # Stage 4: 1/32 resolution (512 channels)
        self.stage1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
        self.stage2 = resnet.layer2
        self.stage3 = resnet.layer3
        self.stage4 = resnet.layer4

    def forward(self, x):
        s1 = self.stage1(x)   # (B, 64,  H/4,  W/4)
        s2 = self.stage2(s1)  # (B, 128, H/8,  W/8)
        s3 = self.stage3(s2)  # (B, 256, H/16, W/16)
        s4 = self.stage4(s3)  # (B, 512, H/32, W/32)
        return s1, s2, s3, s4


class FeaturePyramidNetwork(nn.Module):
    """
    FPN: Fuses multi-scale features from the backbone into a single,
    rich feature map. This allows DBNet to detect both tiny inkjet codes
    AND large printed labels in the same image.
    """
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        # 1x1 convolutions to reduce all stages to the same channel count
        self.lateral4 = nn.Conv2d(in_channels_list[3], out_channels, 1)
        self.lateral3 = nn.Conv2d(in_channels_list[2], out_channels, 1)
        self.lateral2 = nn.Conv2d(in_channels_list[1], out_channels, 1)
        self.lateral1 = nn.Conv2d(in_channels_list[0], out_channels, 1)

        # 3x3 smoothing convolutions after upsampling
        self.smooth4 = nn.Conv2d(out_channels, out_channels // 4, 3, padding=1)
        self.smooth3 = nn.Conv2d(out_channels, out_channels // 4, 3, padding=1)
        self.smooth2 = nn.Conv2d(out_channels, out_channels // 4, 3, padding=1)
        self.smooth1 = nn.Conv2d(out_channels, out_channels // 4, 3, padding=1)

    def forward(self, s1, s2, s3, s4):
        # Top-down pathway: upsample deeper features and add to shallower ones
        p4 = self.lateral4(s4)
        p3 = self.lateral3(s3) + F.interpolate(p4, size=s3.shape[2:], mode='bilinear', align_corners=False)
        p2 = self.lateral2(s2) + F.interpolate(p3, size=s2.shape[2:], mode='bilinear', align_corners=False)
        p1 = self.lateral1(s1) + F.interpolate(p2, size=s1.shape[2:], mode='bilinear', align_corners=False)

        # Smooth and reduce channels
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)
        p1 = self.smooth1(p1)

        # Upsample all to the same size (s1's resolution = H/4, W/4)
        target_size = s1.shape[2:]
        fused = torch.cat([
            p1,
            F.interpolate(p2, size=target_size, mode='bilinear', align_corners=False),
            F.interpolate(p3, size=target_size, mode='bilinear', align_corners=False),
            F.interpolate(p4, size=target_size, mode='bilinear', align_corners=False),
        ], dim=1)  # (B, 256, H/4, W/4)

        return fused


class DBHead(nn.Module):
    """
    DBNet Head: Outputs 2 maps:
    1. Probability Map: "How likely is each pixel to be text?"
    2. Threshold Map: "What is the adaptive threshold at each pixel?"

    The Differentiable Binarization formula combines them:
        Binary Map = 1 / (1 + exp(-k * (P - T)))
    where P is the probability, T is the threshold, and k controls sharpness.
    This is the KEY INNOVATION of DBNet — it makes binarization differentiable
    so the model can learn the optimal threshold during training.
    """
    def __init__(self, in_channels=256, k=50):
        super().__init__()
        self.k = k

        # Probability map head
        self.prob_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, stride=2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, 2, stride=2),
            nn.Sigmoid()
        )

        # Threshold map head (same structure)
        self.thresh_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, stride=2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, 2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, fused_features):
        prob_map = self.prob_head(fused_features)       # (B, 1, H, W)
        thresh_map = self.thresh_head(fused_features)   # (B, 1, H, W)

        # Differentiable Binarization: the core formula
        binary_map = 1.0 / (1.0 + torch.exp(-self.k * (prob_map - thresh_map)))

        return prob_map, thresh_map, binary_map


class DBNet(nn.Module):
    """
    DBNet (Differentiable Binarization Network)
    A lightweight, real-time text detection model.

    Architecture flow:
    Image -> ResNet-18 Backbone -> FPN (multi-scale fusion) -> DB Head
        -> Probability Map (where is the text?)
        -> Threshold Map (adaptive per-pixel threshold)
        -> Binary Map (final crisp text mask via differentiable binarization)
    """
    def __init__(self):
        super().__init__()
        self.backbone = DBNetBackbone()
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[64, 128, 256, 512],
            out_channels=256
        )
        self.head = DBHead(in_channels=256, k=50)

    def forward(self, x):
        s1, s2, s3, s4 = self.backbone(x)
        fused = self.fpn(s1, s2, s3, s4)
        prob_map, thresh_map, binary_map = self.head(fused)
        return prob_map, thresh_map, binary_map


if __name__ == "__main__":
    # Quick test to verify model compiles and runs
    model = DBNet()
    dummy_input = torch.randn(1, 3, 640, 640)
    prob, thresh, binary = model(dummy_input)
    print(f"DBNet compiled successfully!")
    print(f"  Probability Map shape: {prob.shape}  (Expected: [1, 1, 640, 640])")
    print(f"  Threshold Map shape:   {thresh.shape}")
    print(f"  Binary Map shape:      {binary.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,} (Much lighter than Mask R-CNN!)")
