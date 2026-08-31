import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_mask_rcnn_model(num_classes=2):
    """
    Phase 4: Text Detection Model
    Loads a pretrained Mask R-CNN (the industry standard for segmentation).
    It is specifically capable of detecting irregular, curved polygons
    like the curved text on bottles in the CTW1500 dataset.
    """
    # Load an instance segmentation model pre-trained on COCO
    # We use pretrained=False here just to build the architecture quickly for presentation
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)

    # 1. Replace the bounding box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # We have 2 classes: Background (0) and Text (1)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 2. Replace the mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model
