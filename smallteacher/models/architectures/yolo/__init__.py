import torch
from torch import nn
import numpy as np

from pathlib import Path

from .yolo_module import YOLO
from .yolo_config import YOLOConfiguration, generate_config, DEFAULT_CLASSES
from smallteacher.utils import download_from_url
from smallteacher.config import NMS_IOU_THRESHOLD


DEFAULT_WEIGHT_PATH = Path(__file__).parent / "yolov4.weights"
YOLOv4_DARKNET_WEIGHTS = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"


def _load_darknet_weights(network_for_weight_file, new_model, weight_file):
    """Loads weights to layer modules from a pretrained Darknet model.
    One may want to continue training from the pretrained weights, on a dataset with a
    different number of object categories. The number of kernels in the convolutional layers
    just before each detection layer depends on the number of output classes. The Darknet
    solution is to truncate the weight file and stop reading weights at the first incompatible
    layer. For this reason the function silently leaves the rest of the layers unchanged, when
    the weight file ends.
    Args:
        weight_file: A file object containing model weights in the Darknet binary format.
    """
    version = np.fromfile(weight_file, count=3, dtype=np.int32)
    images_seen = np.fromfile(weight_file, count=1, dtype=np.int64)
    print(
        f"Loading weights from Darknet model version {version[0]}.{version[1]}.{version[2]} "
        f"that has been trained on {images_seen[0]} images."
    )

    def read(org_tensor, new_tensor):
        """Reads the contents of ``tensor`` from the current position of ``weight_file``.
        If there's no more data in ``weight_file``, returns without error.
        """
        x = np.fromfile(weight_file, count=org_tensor.numel(), dtype=np.float32)
        if org_tensor.shape != new_tensor.shape:
            print("weight file has incorrect shape for this tensor")
            return
        if x.shape[0] == 0:
            print("No more data!")
            return
        x = torch.from_numpy(x).view_as(new_tensor)
        with torch.no_grad():
            new_tensor.copy_(x)

    for idx, (old_module, new_module) in enumerate(
        zip(network_for_weight_file, new_model.network)
    ):
        assert isinstance(old_module, type(new_module))
        # Weights are loaded only to convolutional layers
        if not isinstance(new_module, nn.Sequential):
            continue

        print(f"Loading weight for {idx}")
        new_conv = new_module[0]
        old_conv = old_module[0]
        assert isinstance(new_conv, nn.Conv2d)
        assert isinstance(old_conv, nn.Conv2d)

        # Convolution may be followed by batch normalization, in which case we read the batch
        # normalization parameters and not the convolution bias.
        if len(new_module) > 1 and isinstance(new_module[1], nn.BatchNorm2d):
            assert isinstance(old_module[1], nn.BatchNorm2d)
            new_bn = new_module[1]
            old_bn = old_module[1]
            read(old_bn.bias, new_bn.bias)
            read(old_bn.weight, new_bn.weight)
            read(old_bn.running_mean, new_bn.running_mean)
            read(old_bn.running_var, new_bn.running_var)
        else:
            read(old_conv.bias, new_conv.bias)

        read(old_conv.weight, new_conv.weight)


def load_darknet_weights(weight_file, model):
    default_model_lines = generate_config(DEFAULT_CLASSES)
    default_model_network = YOLOConfiguration(default_model_lines).get_network()
    _load_darknet_weights(default_model_network, model, weight_file)


def initialize_yolo(
    # YOLO does not consider the background to be a unique class
    num_classes: int,
    pretrained: bool = True,
    # https://github.com/AlexeyAB/darknet/issues/4983
    confidence_threshold: float = 0.005,
    nms_threshold: float = NMS_IOU_THRESHOLD,
    max_predictions_per_image: int = -1,
) -> nn.Module:

    config_lines = generate_config(num_classes)
    model_config = YOLOConfiguration(config_lines)
    model = YOLO(
        network=model_config.get_network(),
        confidence_threshold=confidence_threshold,
        nms_threshold=nms_threshold,
        max_predictions_per_image=max_predictions_per_image,
        image_height=model_config.global_config["height"],
        image_width=model_config.global_config["width"],
    )
    if pretrained:
        if not DEFAULT_WEIGHT_PATH.exists():
            download_from_url(YOLOv4_DARKNET_WEIGHTS, DEFAULT_WEIGHT_PATH)
        with DEFAULT_WEIGHT_PATH.open() as weight_file:
            load_darknet_weights(weight_file, model)
    return model
