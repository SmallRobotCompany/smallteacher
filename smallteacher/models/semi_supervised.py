from torch import nn
import torch
from torchvision.ops import batched_nms
from tqdm import tqdm
import numpy as np
from collections import defaultdict

from collections import OrderedDict
from copy import deepcopy

from .base import FullySupervised
from ..config import (
    LEARNING_RATE,
    DECAY,
    THRESHOLDS_TO_TEST,
    THRESHOLD_OVERESTIMATE,
    FLIP_PROBABILITY,
    BATCH_SIZE,
    INSTANCES_PER_UPDATE,
    NMS_IOU_THRESHOLD,
)
from ..data import LabelledAndUnlabelledDataset
from ..pytorch_augmentations import (
    horizontal_flip,
    vertical_flip,
    deterministic_horizontal_flip,
    deterministic_vertical_flip,
    bbox_hflip,
    bbox_vflip,
)
from ..wbf import weighted_boxes_fusion

from typing import Dict, List, Optional


def no_transform(img, _):
    return img, None


def reverse_no_transform(bbox: torch.Tensor, rows, cols):
    return bbox


def both_flips(img, _):
    return deterministic_horizontal_flip(
        deterministic_vertical_flip(img, None)[0], None
    )


def reverse_both_flips(bbox: torch.Tensor, rows, cols):
    return bbox_vflip(bbox_hflip(bbox, rows, cols), rows, cols)


transform_to_reverse = [
    (deterministic_horizontal_flip, bbox_hflip),
    (deterministic_vertical_flip, bbox_vflip),
    (both_flips, reverse_both_flips),
    (no_transform, reverse_no_transform),
]


class SemiSupervised(FullySupervised):
    def __init__(
        self,
        model_base: str,
        trained_model: Optional[nn.Module] = None,
        learning_rate: float = LEARNING_RATE,
        decay: float = DECAY,
        thresholds: Optional[Dict[str, float]] = None,
        model: Optional[nn.Module] = None,
        use_pseudo_regressions: bool = True,
        ensemble_combination_method: str = "wbf",
        **kwargs,
    ) -> None:
        assert ensemble_combination_method in [
            "nms",
            "wbf",
        ], "Unsupported combination method"
        super().__init__(model_base=model_base, learning_rate=learning_rate, **kwargs)

        if model is not None:
            self.model = model

        if trained_model is not None:
            self.model.load_state_dict(trained_model.state_dict())
        self.teacher = deepcopy(self.model)
        assert self.teacher is not None
        for param in self.teacher.parameters():
            param.detach_()

        self.decay = decay
        self.thresholds = thresholds
        self.use_pseudo_regressions = use_pseudo_regressions
        self._combine = (
            self._combine_preds_nms
            if ensemble_combination_method == "nms"
            else self._combine_preds_wbf
        )

    def on_train_start(self) -> None:
        if self.thresholds is None:
            self.update_thresholds()

    def training_step(self, batch, batch_idx):
        update_every = INSTANCES_PER_UPDATE / BATCH_SIZE

        if batch_idx % update_every == 0:
            self.update_teacher()

        images, targets, unlabelled_images = batch

        loss = 0
        if (not self.use_pseudo_regressions) & (len(images) > 0):
            loss_dict = self.model(images, targets)
            loss += sum(loss for loss in loss_dict.values())
            images, targets = [], []

        if len(unlabelled_images) > 0:
            assert self.teacher is not None
            self.teacher.eval()
            with torch.no_grad():
                preds = []
                for img in unlabelled_images:
                    preds.append(
                        self._combine(
                            self.weakly_ensembled_teacher_forward(img),
                            img.shape,
                            self.thresholds,
                        )
                    )
                for img, pred in zip(unlabelled_images, preds):
                    img, pred = horizontal_flip(img, pred, FLIP_PROBABILITY)
                    img, pred = vertical_flip(img, pred, FLIP_PROBABILITY)
                    images.append(img)
                    targets.append(pred)

        if len(images) > 0:
            # len(images) might be 0 if we are not using pseudo regressions
            # and there are no unlabelled images in the batch
            loss_dict = self.model(images, targets)
            if not self.use_pseudo_regressions:
                loss += sum(
                    loss for key, loss in loss_dict.items() if "box_reg" not in key
                )
            else:
                loss += sum(loss for loss in loss_dict.values())

        self.log("training_loss", loss)
        return loss

    def on_train_epoch_end(self) -> None:
        self.update_teacher()
        self.update_thresholds()

    @torch.no_grad()
    def update_teacher(self):
        # https://gist.github.com/zijian-hu/cb2224cca05565cc104e1da379380488
        model_params = OrderedDict(self.model.named_parameters())
        teacher_params = OrderedDict(self.teacher.named_parameters())

        # check if both model contains the same set of keys
        assert model_params.keys() == teacher_params.keys()

        for name, param in model_params.items():
            # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            # shadow_variable -= (1 - decay) * (shadow_variable - variable)
            teacher_params[name].sub_(
                (1.0 - self.decay) * (teacher_params[name] - param)
            )

        model_buffers = OrderedDict(self.model.named_buffers())
        teacher_buffers = OrderedDict(self.teacher.named_buffers())

        # check if both model contains the same set of keys
        assert model_buffers.keys() == teacher_buffers.keys()

        for name, buffer in model_buffers.items():
            # buffers are copied
            teacher_buffers[name].copy_(buffer)

    def _find_threshold(
        self,
        idx: int,
        real_distribution: np.ndarray,
        predictions_per_class: List[List[Dict]],
        image_sizes: List[torch.Size],
    ) -> float:
        real_predictions_per_image = real_distribution.sum() / len(real_distribution)
        with_overestimate = real_predictions_per_image * THRESHOLD_OVERESTIMATE
        thresholds, diff_vals = [], []
        preds_per_image = []
        for threshold in THRESHOLDS_TO_TEST:
            preds = [
                self._combine(pred, im, {idx: threshold})
                for pred, im in zip(predictions_per_class, image_sizes)
            ]
            preds_in_image = [len(pred["scores"]) for pred in preds]
            num_predictions_per_image = sum(preds_in_image) / len(preds_in_image)
            diff_vals.append(abs(num_predictions_per_image - with_overestimate))
            thresholds.append(threshold)
            preds_per_image.append(num_predictions_per_image)
        print(real_predictions_per_image, with_overestimate, preds_per_image)
        return thresholds[diff_vals.index(min(diff_vals))]

    @torch.no_grad()
    def update_thresholds(self) -> None:
        print("Updating thresholds")
        assert isinstance(
            self.trainer.train_dataloader.dataset.datasets, LabelledAndUnlabelledDataset
        )
        predictions_per_class: Dict[int, List] = defaultdict(list)
        image_sizes = []
        self.teacher.eval()
        for image in tqdm(
            self.trainer.train_dataloader.dataset.datasets.random_unlabelled_sample(),
        ):
            predictions = self.weakly_ensembled_teacher_forward(image.to(self.device))
            for idx in self.trainer.train_dataloader.dataset.datasets.classes:

                predictions_per_class[idx].append(
                    [
                        {
                            key: val[sub_preds["labels"] == idx]
                            for key, val in sub_preds.items()
                        }
                        for sub_preds in predictions
                    ]
                )
            image_sizes.append(image.shape)

        self.thresholds = {
            idx: self._find_threshold(
                idx,
                self.trainer.train_dataloader.dataset.datasets.distribution_for_class(
                    idx
                ),
                predictions_per_class[idx],
                image_sizes,
            )
            for idx in predictions_per_class.keys()
        }

        print(self.thresholds)
        for key, val in self.thresholds.items():
            self.log(f"{key}_threshold", val)

    @torch.no_grad()
    def _create_mask(
        self, labels: torch.Tensor, scores: torch.Tensor, thresholds: Dict
    ) -> torch.Tensor:

        all_idxs = list(thresholds.keys())
        mask = (labels == all_idxs[0]) & (scores >= thresholds[all_idxs[0]])
        for idx in all_idxs[1:]:
            mask |= (labels == idx) & (scores >= thresholds[idx])
        return mask

    @torch.no_grad()
    def filter_target(self, predictions: Dict, thresholds: Dict) -> Dict:
        target = predictions
        boxes_np = target["boxes"]
        labels_np = target["labels"]
        scores_np = target["scores"]
        mask = self._create_mask(labels_np, scores_np, thresholds)

        return {"boxes": boxes_np[mask], "labels": labels_np[mask]}

    def _combine_preds_nms(
        self, preds: List, img_shape: torch.Size, thresholds: Dict
    ) -> Dict:
        combined_preds = preds[0]
        for other_pred in preds[1:]:
            for key in combined_preds.keys():
                combined_preds[key] = torch.cat([combined_preds[key], other_pred[key]])
        indices_to_keep = batched_nms(
            combined_preds["boxes"],
            combined_preds["scores"],
            combined_preds["labels"],
            iou_threshold=NMS_IOU_THRESHOLD,
        )
        output = {key: val[indices_to_keep] for key, val in combined_preds.items()}
        return self.filter_target(output, thresholds)

    def _combine_preds_wbf(
        self, preds: List, img_shape: torch.Size, thresholds: Dict
    ) -> Dict:
        def normalize_boxes(boxes: torch.Tensor) -> List:
            return (
                boxes.cpu()
                / torch.tensor([img_shape[2], img_shape[1], img_shape[2], img_shape[1]])
            ).tolist()

        def denormalize_boxes(boxes: List):
            return (
                torch.tensor(boxes)
                * torch.tensor([img_shape[2], img_shape[1], img_shape[2], img_shape[1]])
            ).to(self.device)

        boxes_list = [normalize_boxes(pred["boxes"]) for pred in preds]
        scores_list = [pred["scores"].cpu().tolist() for pred in preds]
        labels_list = [pred["labels"].cpu().tolist() for pred in preds]
        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            iou_thr=NMS_IOU_THRESHOLD,
            skip_box_thr=thresholds,
        )

        return {
            "boxes": denormalize_boxes(boxes),
            "scores": torch.tensor(scores, device=self.device),
            "labels": torch.tensor(labels, device=self.device).to(torch.int64),
        }

    @torch.no_grad()
    def weakly_ensembled_teacher_forward(self, img: torch.Tensor) -> Dict:
        combined_preds = []
        for transform, reverse in transform_to_reverse:
            pred = self.teacher([transform(img, None)[0]])[0]
            pred["boxes"] = reverse(pred["boxes"], img.shape[1], img.shape[2])
            combined_preds.append(pred)
        return combined_preds
