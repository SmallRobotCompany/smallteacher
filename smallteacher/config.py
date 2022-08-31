import torch


BEST_MODEL_NAME = "best_model-{epoch}"

BATCH_SIZE = 2

MAX_PSUEDO_LABELLED_IMAGES = 2000
INSTANCES_PER_UPDATE = 64
LEARNING_RATE = 1e-5
DECAY = 0.9996
THRESHOLDS_TO_TEST = [x / 10 for x in range(1, 10)]
THRESHOLD_OVERESTIMATE = 1 + 1 / 2

FLIP_PROBABILITY = 0.5
HUE = 0.1
SATURATIONS = [1 / 1.5, 1.5]
CONTRASTS = [1 / 1.5, 1.5]
NMS_IOU_THRESHOLD = 0.7

TEST_MAP_KWARGS = {
    "max_detection_thresholds": [100],
    "class_metrics": True,
    "iou_thresholds": torch.linspace(
        0.5, 0.95, round((0.95 - 0.5) / 0.05) + 1
    ).tolist(),
    "rec_thresholds": torch.linspace(0.0, 1.00, round(1.00 / 0.01) + 1).tolist(),
}
