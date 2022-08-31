from torch import nn

from typing import Any

from torchvision.models.detection.ssd import ssd300_vgg16

from typing import Optional


def initialize_ssd(
    num_classes: int,
    pretrained: bool = False,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: Optional[int] = 5,
    **kwargs: Any,
) -> nn.Module:
    return ssd300_vgg16(
        pretrained=pretrained,
        trainable_backbone_layers=trainable_backbone_layers,
        # add 1 since SSD considers the background to be a class too
        num_classes=num_classes + 1,
        pretrained_backbone=pretrained_backbone,
        **kwargs,
    )
