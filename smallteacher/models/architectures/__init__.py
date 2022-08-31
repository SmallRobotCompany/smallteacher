from .faster_rcnn import initialize_fasterrcnn
from .retinanet import initialize_retinanet
from .ssd import initialize_ssd
from .yolo import initialize_yolo


STR2FUNC = {
    "FRCNN": initialize_fasterrcnn,
    "RetinaNet": initialize_retinanet,
    "SSD": initialize_ssd,
    "YOLO": initialize_yolo,
}
