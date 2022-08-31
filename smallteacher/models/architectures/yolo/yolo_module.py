import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from .yolo_layers import DetectionLayer, RouteLayer, ShortcutLayer
from ....config import NMS_IOU_THRESHOLD

from torchvision.ops import nms

log = logging.getLogger(__name__)


class YOLO(nn.Module):
    """PyTorch Lightning implementation of YOLOv3 and YOLOv4.
    *YOLOv3 paper*: `Joseph Redmon and Ali Farhadi <https://arxiv.org/abs/1804.02767>`_
    *YOLOv4 paper*: `Alexey Bochkovskiy, Chien-Yao Wang, and Hong-Yuan Mark Liao <https://arxiv.org/abs/2004.10934>`_
    *Implementation*: `Seppo Enarvi <https://github.com/senarvi>`_
    The network architecture can be read from a Darknet configuration file using the
    :class:`~pl_bolts.models.detection.yolo.yolo_config.YOLOConfiguration` class, or created by
    some other means, and provided as a list of PyTorch modules.
    The input from the data loader is expected to be a list of images. Each image is a tensor with
    shape ``[channels, height, width]``. The images from a single batch will be stacked into a
    single tensor, so the sizes have to match. Different batches can have different image sizes, as
    long as the size is divisible by the ratio in which the network downsamples the input.
    During training, the model expects both the input tensors and a list of targets. *Each target is
    a dictionary containing*:
    - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in `(x1, y1, x2, y2)` format
    - labels (``Int64Tensor[N]``): the class label for each ground-truth box
    :func:`~pl_bolts.models.detection.yolo.yolo_module.YOLO.forward` method returns all
    predictions from all detection layers in all images in one tensor with shape
    ``[images, predictors, classes + 5]``. The coordinates are scaled to the input image size.
    During training it also returns a dictionary containing the classification, box overlap, and
    confidence losses.
    During inference, the model requires only the input tensors.
    :func:`~pl_bolts.models.detection.yolo.yolo_module.YOLO.infer` method filters and processes the
    predictions. *The processed output includes the following tensors*:
    - boxes (``FloatTensor[N, 4]``): predicted bounding box `(x1, y1, x2, y2)` coordinates in image space
    - scores (``FloatTensor[N]``): detection confidences
    - labels (``Int64Tensor[N]``): the predicted labels for each image
    Weights can be loaded from a Darknet model file using ``load_darknet_weights()``.
    CLI command::
        # PascalVOC
        wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny-3l.cfg
        python yolo_module.py --config yolov4-tiny-3l.cfg --data_dir . --gpus 8 --batch_size 8
    """

    def __init__(
        self,
        network: nn.ModuleList,
        confidence_threshold: float = 0.2,
        nms_threshold: float = NMS_IOU_THRESHOLD,
        max_predictions_per_image: int = -1,
        image_height: int = 576,
        image_width: int = 576,
    ) -> None:
        """
        Args:
            network: A list of network modules. This can be obtained from a Darknet configuration
                using the :func:`~pl_bolts.models.detection.yolo.yolo_config.YOLOConfiguration.get_network`
                method.
            optimizer: Which optimizer class to use for training.
            optimizer_params: Parameters to pass to the optimizer constructor.
            lr_scheduler: Which learning rate scheduler class to use for training.
            lr_scheduler_params: Parameters to pass to the learning rate scheduler constructor.
            confidence_threshold: Postprocessing will remove bounding boxes whose
                confidence score is not higher than this threshold.
            nms_threshold: Non-maximum suppression will remove bounding boxes whose IoU with a higher
                confidence box is higher than this threshold, if the predicted categories are equal.
            max_predictions_per_image: If non-negative, keep at most this number of
                highest-confidence predictions per image.
        """
        super().__init__()

        self.network = network
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_predictions_per_image = max_predictions_per_image

        self.transform = GeneralizedRCNNTransform(
            min_size=min([image_width, image_height]),
            max_size=max(image_width, image_height),
            image_mean=[0, 0, 0],
            image_std=[1, 1, 1],
            fixed_size=(image_width, image_height),
        )

    def update_confidence_threshold(self, new_threshold: float) -> None:
        # at training time, YOLO uses a confidence threshold of 0.005
        # to calculate mAP. Users may want to update this
        # for inference
        self.confidence_threshold = new_threshold

    def forward(
        self,
        images: Union[List[Tensor], Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Runs a forward pass through the network (all layers listed in ``self.network``), and if training targets
        are provided, computes the losses from the detection layers.
        Detections are concatenated from the detection layers. Each image will produce
        `N * num_anchors * grid_height * grid_width` detections, where `N` depends on the number of
        detection layers. For one detection layer `N = 1`, and each detection layer increases it by
        a number that depends on the size of the feature map on that layer. For example, if the
        feature map is twice as wide and high as the grid, the layer will add four times more
        features.
        Args:
            images: Images to be processed. Tensor of size
                ``[batch_size, num_channels, height, width]``.
            targets: If set, computes losses from detection layers against these targets. A list of
                dictionaries, one for each image.
        Returns:
            detections (:class:`~torch.Tensor`), losses (Dict[str, :class:`~torch.Tensor`]):
            Detections, and if targets were provided, a dictionary of losses. Detections are shaped
            ``[batch_size, num_predictors, num_classes + 5]``, where ``num_predictors`` is the
            total number of cells in all detection layers times the number of boxes predicted by
            one cell. The predicted box coordinates are in `(x1, y1, x2, y2)` format and scaled to
            the input image size.
        """
        if targets is not None:
            for target in targets:
                target["labels"] = target["labels"].detach().clone() - 1
        if isinstance(images, (list, tuple)):
            images, targets = self._validate_batch((images, targets))
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        image_list, targets = self.transform(images, targets)
        outputs = []  # Outputs from all layers
        detections = []  # Outputs from detection layers
        losses = []  # Losses from detection layers
        hits = []  # Number of targets each detection layer was responsible for

        image_height = image_list.tensors.shape[2]
        image_width = image_list.tensors.shape[3]
        image_size = torch.tensor(
            [image_width, image_height], device=image_list.tensors.device
        )

        x = image_list.tensors
        for module in self.network:
            if isinstance(module, (RouteLayer, ShortcutLayer)):
                x = module(x, outputs)
            elif isinstance(module, DetectionLayer):
                if targets is None:
                    x = module(x, image_size)
                    detections.append(x)
                else:
                    x, layer_losses, layer_hits = module(x, image_size, targets)
                    detections.append(x)
                    losses.append(layer_losses)
                    hits.append(layer_hits)
            else:
                x = module(x)

            outputs.append(x)

        if targets is not None:
            total_hits = sum(hits)
            num_targets = sum(len(image_targets["boxes"]) for image_targets in targets)
            if total_hits != num_targets:
                log.warning(
                    f"{num_targets} training targets were matched a total of {total_hits} times by detection layers. "
                    "Anchors may have been configured incorrectly."
                )

            def total_loss(loss_name):
                """Returns the sum of the loss over detection layers."""
                loss_tuple = tuple(layer_losses[loss_name] for layer_losses in losses)
                return torch.stack(loss_tuple).sum()

            losses = {
                loss_name: total_loss(loss_name) for loss_name in losses[0].keys()
            }
            return losses

        detections = torch.cat(detections, 1)
        detections = self._split_detections(detections)
        detections = self._filter_detections(detections)
        detections = self._split_detections_again(detections)
        detections = self.transform.postprocess(
            detections, image_list.image_sizes, original_image_sizes
        )
        return detections

    def _validate_batch(
        self, batch: Tuple[List[Tensor], Optional[List[Dict[str, Tensor]]]]
    ) -> Tuple[Tensor, List[Dict[str, Tensor]]]:
        """Reads a batch of data, validates the format
        Args:
            batch: The batch of data read by the :class:`~torch.utils.data.DataLoader`.
        Returns:
            The input batch with images stacked into a single tensor.
        """
        images, targets = batch

        for image in images:
            if not isinstance(image, Tensor):
                raise ValueError(
                    f"Expected image to be of type Tensor, got {type(image)}."
                )

        if targets is not None:

            if len(images) != len(targets):
                raise ValueError(
                    f"Got {len(images)} images, but targets for {len(targets)} images."
                )

            for target in targets:
                boxes = target["boxes"]
                if not isinstance(boxes, Tensor):
                    raise ValueError(
                        f"Expected target boxes to be of type Tensor, got {type(boxes)}."
                    )
                if (len(boxes.shape) != 2) or (boxes.shape[-1] != 4):
                    raise ValueError(
                        f"Expected target boxes to be tensors of shape [N, 4], got {list(boxes.shape)}."
                    )
                labels = target["labels"]
                if not isinstance(labels, Tensor):
                    raise ValueError(
                        f"Expected target labels to be of type Tensor, got {type(labels)}."
                    )
                if len(labels.shape) != 1:
                    raise ValueError(
                        f"Expected target labels to be tensors of shape [N], got {list(labels.shape)}."
                    )
        return images, targets

    @staticmethod
    def _split_detections(detections: Tensor) -> Dict[str, Tensor]:
        """Splits the detection tensor returned by a forward pass into a dictionary.
        The fields of the dictionary are as follows:
            - boxes (``Tensor[batch_size, N, 4]``): detected bounding box `(x1, y1, x2, y2)` coordinates
            - scores (``Tensor[batch_size, N]``): detection confidences
            - classprobs (``Tensor[batch_size, N]``): probabilities of the best classes
            - labels (``Int64Tensor[batch_size, N]``): the predicted labels for each image
        Args:
            detections: A tensor of detected bounding boxes and their attributes.
        Returns:
            A dictionary of detection results.
        """
        boxes = detections[..., :4]
        scores = detections[..., 4]
        classprobs = detections[..., 5:]
        classprobs, labels = torch.max(classprobs, -1)
        return {
            "boxes": boxes,
            "scores": scores,
            "classprobs": classprobs,
            "labels": labels + 1,
        }

    @staticmethod
    def _split_detections_again(
        detections: Dict[str, Tensor]
    ) -> List[Dict[str, List[Tensor]]]:
        """Splits the detection tensor returned by a forward pass into a list of dictionaries.
        The fields of each dictionary are as follows:
            - boxes (``Tensor[N, 4]``): detected bounding box `(x1, y1, x2, y2)` coordinates
            - scores (``Tensor[N]``): detection confidences
            - classprobs (``Tensor[N]``): probabilities of the best classes
            - labels (``Int64Tensor[N]``): the predicted labels for each image
        Args:
            detections: A tensor of detected bounding boxes and their attributes.
        Returns:
            A dictionary of detection results.
        """
        num_instances = len(detections["boxes"])
        output_detections = []
        for i in range(num_instances):
            output_detections.append(
                {key: detections[key][i] for key in detections.keys()}
            )
        return output_detections

    def _filter_detections(
        self, detections: Dict[str, Tensor]
    ) -> Dict[str, List[Tensor]]:
        """Filters detections based on confidence threshold. Then for every class performs non-maximum suppression
        (NMS). NMS iterates the bounding boxes that predict this class in descending order of confidence score, and
        removes lower scoring boxes that have an IoU greater than the NMS threshold with a higher scoring box.
        Finally the detections are sorted by descending confidence and possible truncated to the maximum number of
        predictions.
        Args:
            detections: All detections. A dictionary of tensors, each containing the predictions
                from all images.
        Returns:
            Filtered detections. A dictionary of lists, each containing a tensor per image.
        """
        boxes = detections["boxes"]
        scores = detections["scores"]
        classprobs = detections["classprobs"]
        labels = detections["labels"]

        out_boxes = []
        out_scores = []
        out_classprobs = []
        out_labels = []

        for img_boxes, img_scores, img_classprobs, img_labels in zip(
            boxes, scores, classprobs, labels
        ):
            # Select detections with high confidence score.
            selected = img_scores > self.confidence_threshold
            img_boxes = img_boxes[selected]
            img_scores = img_scores[selected]
            img_classprobs = img_classprobs[selected]
            img_labels = img_labels[selected]

            img_out_boxes = boxes.new_zeros((0, 4))
            img_out_scores = scores.new_zeros(0)
            img_out_classprobs = classprobs.new_zeros(0)
            img_out_labels = labels.new_zeros(0)

            # Iterate through the unique object classes detected in the image and perform non-maximum
            # suppression for the objects of the class in question.
            for cls_label in labels.unique():
                selected = img_labels == cls_label
                cls_boxes = img_boxes[selected]
                cls_scores = img_scores[selected]
                cls_classprobs = img_classprobs[selected]
                cls_labels = img_labels[selected]

                # NMS will crash if there are too many boxes.
                cls_boxes = cls_boxes[:100000]
                cls_scores = cls_scores[:100000]
                selected = nms(cls_boxes, cls_scores, self.nms_threshold)

                img_out_boxes = torch.cat((img_out_boxes, cls_boxes[selected]))
                img_out_scores = torch.cat((img_out_scores, cls_scores[selected]))
                img_out_classprobs = torch.cat(
                    (img_out_classprobs, cls_classprobs[selected])
                )
                img_out_labels = torch.cat((img_out_labels, cls_labels[selected]))

            # Sort by descending confidence and limit the maximum number of predictions.
            indices = torch.argsort(img_out_scores, descending=True)
            if self.max_predictions_per_image >= 0:
                indices = indices[: self.max_predictions_per_image]
            out_boxes.append(img_out_boxes[indices])
            out_scores.append(img_out_scores[indices])
            out_classprobs.append(img_out_classprobs[indices])
            out_labels.append(img_out_labels[indices])

        return {
            "boxes": out_boxes,
            "scores": out_scores,
            "classprobs": out_classprobs,
            "labels": out_labels,
        }
