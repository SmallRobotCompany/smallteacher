from smallteacher.models.architectures.yolo.yolo_config import (
    YOLOConfiguration,
    generate_config,
    DEFAULT_CLASSES,
)
from smallteacher.models.architectures.yolo import initialize_yolo


def test_yolo_config_with_new_classes():

    for classes in [DEFAULT_CLASSES - 1, DEFAULT_CLASSES, DEFAULT_CLASSES + 1]:
        config = generate_config(classes)

        model_config = YOLOConfiguration(config)
        _ = model_config.get_network()


def test_yolo_init():
    _ = initialize_yolo(num_classes=2, pretrained=False)
