"""
model.py - Mask R-CNN with improved backbone (ResNet-50/101 + FPN)
Modifications:
  1. ResNet-101 FPN backbone (stronger than default ResNet-50)
  2. Larger anchor sizes tuned for cell images
  3. Increased RPN and RoI sampling for better small-object recall
  4. Test-time augmentation (optional)

Reference:
  He et al., "Mask R-CNN", ICCV 2017
  Lin et al., "Feature Pyramid Networks for Object Detection", CVPR 2017
"""

import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch

NUM_CLASSES = 5  # 4 cell classes + 1 background


def build_model(num_classes=NUM_CLASSES, backbone_name="resnet50", pretrained=True):
    """
    Build Mask R-CNN with ResNet-50 or ResNet-101 FPN backbone.

    Key design choices:
    - FPN backbone for multi-scale feature extraction (critical for cells of varying sizes)
    - Custom anchor sizes: smaller anchors to better capture small cells
    - Higher NMS thresh during training for denser predictions
    - Larger RoI batch size for more mask supervision per step
    """

    # ---- Backbone ----
    # ResNet-101 gives ~1-2 AP improvement over ResNet-50 for dense prediction tasks
    weights_name = (
        "ResNet50_Weights.IMAGENET1K_V1"
        if backbone_name == "resnet50"
        else "ResNet101_Weights.IMAGENET1K_V1"
    )
    backbone = resnet_fpn_backbone(
        backbone_name=backbone_name,
        weights=weights_name if pretrained else None,
        trainable_layers=3,  # Unfreeze last 3 layers (layer2, layer3, layer4)
    )

    # ---- Anchor Generator ----
    # Default anchors are too large for small cells.
    # We use smaller base sizes and keep 3 aspect ratios.
    anchor_generator = AnchorGenerator(
        sizes=((8,), (16,), (32,), (64,), (128,)),  # one per FPN level
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )

    # ---- RoI Align pooling output size ----
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=7,
        sampling_ratio=2,
    )
    mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=14,
        sampling_ratio=2,
    )

    # ---- Assemble model ----
    model = MaskRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        mask_roi_pool=mask_roi_pooler,
        # RPN settings
        rpn_pre_nms_top_n_train=4000,  # 原為 3000
        rpn_pre_nms_top_n_test=2000,  # 原為 1000
        rpn_post_nms_top_n_train=3000,  # 原為 2000
        rpn_post_nms_top_n_test=1000,  # 原為 500
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        # Box head settings
        box_batch_size_per_image=256,
        box_positive_fraction=0.25,
        box_nms_thresh=0.5,
        box_score_thresh=0.05,  # low threshold at inference; filter later
        box_detections_per_img=500,  # 原為 300  # more detections per image for dense cells
    )

    return model


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total/1e6:.1f}M  |  Trainable: {trainable/1e6:.1f}M")
    return trainable


if __name__ == "__main__":
    model = build_model(backbone_name="resnet50")
    count_parameters(model)
    model.eval()
    dummy = [torch.rand(3, 512, 512)]
    with torch.no_grad():
        out = model(dummy)
    print("Output keys:", out[0].keys())
