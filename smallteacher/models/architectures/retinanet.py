from torch import nn

from typing import Any

from torchvision.models.detection.retinanet import RetinaNetHead, retinanet_resnet50_fpn

from typing import Optional


def initialize_retinanet(
    num_classes: int,
    pretrained: bool = False,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: Optional[int] = 5,
    **kwargs: Any,
) -> nn.Module:
    model = retinanet_resnet50_fpn(
        pretrained=pretrained,
        pretrained_backbone=pretrained_backbone,
        trainable_backbone_layers=trainable_backbone_layers,
        **kwargs,
    )

    model.head = RetinaNetHead(
        in_channels=model.backbone.out_channels,
        num_anchors=model.head.classification_head.num_anchors,
        # add 1 since RetinaNet considers the background to be a class too
        num_classes=num_classes + 1,
        **kwargs,
    )
    return model
