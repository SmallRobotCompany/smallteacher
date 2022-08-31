import torch
from pytorch_lightning import LightningModule
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import numpy as np
from datetime import datetime

from .iou import evaluate_iou
from ..constants import Metrics
from ..config import TEST_MAP_KWARGS, LEARNING_RATE

from .architectures import STR2FUNC

from typing import Dict, List, Optional


class DetectionBase(LightningModule):
    def __init__(
        self,
        model_base: Optional[str],
        num_classes: int,
        learning_rate: float = LEARNING_RATE,
        **kwargs,
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        if model_base is not None:
            self.model = STR2FUNC[model_base](num_classes=num_classes, **kwargs)

    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, _):
        images, targets = batch
        # fasterrcnn takes only images for eval() mode
        with torch.no_grad():
            self.model.eval()
            outs = self.model(images)
        iou = torch.stack([evaluate_iou(t, o) for t, o in zip(targets, outs)]).mean()
        return {Metrics.IOU: iou, Metrics.PREDICTIONS: outs, Metrics.TARGETS: targets}

    def test_epoch_end(self, outs):
        metric = MeanAveragePrecision(**TEST_MAP_KWARGS)
        for output in outs:
            metric.update(
                self._to_cpu(output[Metrics.PREDICTIONS]),
                self._to_cpu(output[Metrics.TARGETS]),
            )
        computed_metrics = metric.compute()
        print(f"mAP: {computed_metrics['map']}, \n {computed_metrics}")

        # now, we also want to plot precision recall curves.
        classes = metric._get_classes()
        precisions, _ = metric._calculate(classes)
        # the precision tensors have the following dimensions:
        # [IOU_THRESHOLD, RECALL_THRESHOLD, CLASS_IDX, BBOX_SIZE, MAX_DETECTION]
        # we care about all bbox sizes (index 0) and only use a single max detection
        # value (index 0), so we want to plot precision and recall by
        # iterating through the recall threshold for a given class index
        precisions = precisions[:, :, :, 0, 0].detach().cpu().numpy()

        # This is hacky because pytorch lightning makes it super difficult to
        # nicely save stuff in the checkpoint folder >:(
        precision_filename = (
            f"{str(datetime.now()).replace(' ', 'H')}-precision_curve.npy"
        )
        np.save(precision_filename, precisions)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )
        return {
            "optimizer": optimizer,
        }


class FullySupervised(DetectionBase):
    def training_step(self, batch, _):
        images, targets = batch
        # fasterrcnn takes both images and targets for training, returns
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log("training_loss", loss)
        return {"loss": loss, "log": loss_dict}

    def validation_step(self, batch, _):
        images, targets = batch
        # fasterrcnn takes only images for eval() mode
        with torch.no_grad():
            self.model.eval()
            outs = self.model(images)
        iou = torch.stack([evaluate_iou(t, o) for t, o in zip(targets, outs)]).mean()
        return {Metrics.IOU: iou, Metrics.PREDICTIONS: outs, Metrics.TARGETS: targets}

    def validation_epoch_end(self, outs):
        avg_iou = torch.stack([o[Metrics.IOU] for o in outs]).mean()
        self.log(Metrics.AVG_IOU, avg_iou)

        metric = MeanAveragePrecision()
        for output in outs:
            metric.update(
                self._to_cpu(output[Metrics.PREDICTIONS]),
                self._to_cpu(output[Metrics.TARGETS]),
            )
        computed_metrics = metric.compute()
        self.log(Metrics.MAP, computed_metrics["map"])
        return {
            Metrics.AVG_IOU: avg_iou,
            "log": {Metrics.AVG_IOU: avg_iou, Metrics.MAP: computed_metrics["map"]},
        }

    @staticmethod
    def _to_cpu(
        targets: List[Dict[str, torch.Tensor]]
    ) -> List[Dict[str, torch.Tensor]]:
        return [{key: val.detach().cpu() for key, val in d.items()} for d in targets]
