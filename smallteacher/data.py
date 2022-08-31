from random import choice, sample, uniform
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from itertools import accumulate

import torch
from torch import minimum, maximum, ones_like
from torch.utils.data import DataLoader
from torchvision.transforms import ColorJitter
from pytorch_lightning import LightningDataModule

from .pytorch_augmentations import horizontal_flip, vertical_flip
from .config import (
    MAX_PSUEDO_LABELLED_IMAGES,
    BATCH_SIZE,
    FLIP_PROBABILITY,
    HUE,
    SATURATIONS,
    CONTRASTS,
)

from typing import Any, Dict, Generator, List, Optional, Tuple

SHARING_STRATEGY = "file_system"

torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)


def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)


def train_augmentations(img, target):
    img = ColorJitter(hue=HUE, saturation=SATURATIONS, contrast=CONTRASTS)(img)
    img, target = horizontal_flip(img, target, FLIP_PROBABILITY)
    return vertical_flip(img, target, FLIP_PROBABILITY)


class LabelledAndUnlabelledDataset:
    def __init__(
        self,
        labelled_dataset,
        unlabelled_dataset,
        unlabelled_images_per_epoch: Optional[int] = MAX_PSUEDO_LABELLED_IMAGES,
    ) -> None:

        self.labelled_dataset = labelled_dataset
        self.unlabelled_dataset = unlabelled_dataset

        self.unlabelled_images_per_epoch = unlabelled_images_per_epoch
        self.mapper = []
        if unlabelled_images_per_epoch is not None:
            self.mapper = list(
                range(int(len(unlabelled_dataset) / unlabelled_images_per_epoch))
            )
        self._distribution_per_class: Dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        if self.unlabelled_images_per_epoch is None:
            return len(self.labelled_dataset) + len(self.unlabelled_dataset)
        else:
            return len(self.labelled_dataset) + self.unlabelled_images_per_epoch

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if idx < len(self.labelled_dataset):
            return *self.labelled_dataset[idx], None
        else:
            unlab_idx = idx - len(self.labelled_dataset)
            if self.unlabelled_images_per_epoch is not None:
                unlab_idx = unlab_idx + (
                    self.unlabelled_images_per_epoch * choice(self.mapper)
                )
            return None, None, self.unlabelled_dataset[unlab_idx]

    @staticmethod
    def collate_fn(batch):
        images, targets, unlab_images = tuple(zip(*batch))

        images = [im for im in images if im is not None]
        targets = [t for t in targets if t is not None]
        assert len(images) == len(targets)

        unlab_images = [un_im for un_im in unlab_images if un_im is not None]

        return images, targets, unlab_images

    def _calculate_class_distributions(self) -> None:
        print("Calculating real distributions per class")
        # we set the max to 1 because 0 is the background class,
        # so we will ignore it
        max_label_idx = 1
        num_per_class = defaultdict(list)
        for idx in tqdm(range(len(self.labelled_dataset))):
            _, targets = self.labelled_dataset[idx]
            if max(targets["labels"]) > max_label_idx:
                # fill in all the missing values with 0s
                max_label_idx = max(targets["labels"])
                num_filled_values = len(num_per_class[0])
                for label_idx in range(1, max_label_idx + 1):
                    if len(num_per_class[label_idx]) < num_filled_values:
                        missing = num_filled_values - len(num_per_class[label_idx])
                        num_per_class[label_idx].extend([0] * missing)

            for label_idx in range(1, max_label_idx + 1):
                num_per_class[label_idx].append(sum(targets["labels"] == label_idx))

        self._distribution_per_class = {
            key: np.array(val) for key, val in num_per_class.items() if key != 0
        }

    def distribution_for_class(self, class_idx: int) -> torch.Tensor:
        if len(self._distribution_per_class) == 0:
            self._calculate_class_distributions()

        return self._distribution_per_class[class_idx]

    @property
    def classes(self) -> List[int]:
        if len(self._distribution_per_class) == 0:
            self._calculate_class_distributions()
        return list(self._distribution_per_class.keys())

    def random_unlabelled_sample(
        self,
    ) -> Generator[torch.Tensor, None, None]:
        indices_to_sample = sample(
            list(range(len(self.unlabelled_dataset))), k=len(self.labelled_dataset)
        )
        for idx in indices_to_sample:
            yield self.unlabelled_dataset[idx]


class ListOfDatasetsWithMosaicing:
    def __init__(self, datasets, mosaic: bool = False):
        if not isinstance(datasets, list):
            datasets = [datasets]
        self.datasets = [
            self._check_dataset(dataset, datasets[0][0]) for dataset in datasets
        ]
        self.lengths = [len(d) for d in self.datasets]
        if mosaic:
            assert self._is_labelled_dataset
        self._mosaic = mosaic

    @staticmethod
    def _check_dataset(dataset, expected_output):
        assert hasattr(dataset, "__getitem__")
        assert hasattr(dataset, "__len__")
        if len(dataset) > 0:
            dataset_output = dataset[0]
            assert isinstance(dataset_output, type(expected_output))
            if isinstance(dataset_output, tuple):
                assert len(dataset_output) == len(expected_output)
                for i in range(len(dataset_output)):
                    assert isinstance(dataset_output[i], type(expected_output[i]))
        return dataset

    @property
    def _is_labelled_dataset(self) -> bool:
        output = self.datasets[0][0]
        if isinstance(output, tuple):
            if len(output) == 2:
                if isinstance(output[1], dict):
                    return True
        return False

    def random_sample(self, k: int) -> Generator[Any, None, None]:
        for _ in range(k):
            dataset = choice(self.datasets)
            index = choice(list(range(len(dataset))))
            yield dataset[index]

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        cur_min = 0
        for dataset_idx, cur_max in enumerate(accumulate(self.lengths)):
            if (idx >= cur_min) & (idx < cur_max):
                output = self.datasets[dataset_idx][idx - cur_min]
                if self._mosaic:
                    return self.mosaic(*output)
                return output
            cur_min = cur_max

    @staticmethod
    def _box_sizes(boxes: torch.Tensor) -> torch.Tensor:
        x_len = maximum((boxes[:, 3] - boxes[:, 1]), torch.zeros(boxes.shape[0]))
        y_len = maximum((boxes[:, 2] - boxes[:, 0]), torch.zeros(boxes.shape[0]))
        return x_len * y_len

    @classmethod
    def _trim_annotations(
        cls,
        anns: Dict,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        min_size: float = 0.5,
    ):
        # boxes are [x_min, y_min, x_max, y_max]
        new_boxes = anns["boxes"].clone()
        new_boxes[:, 0] = maximum(new_boxes[:, 0], ones_like(new_boxes[:, 0]) * x_min)
        new_boxes[:, 1] = maximum(new_boxes[:, 1], ones_like(new_boxes[:, 1]) * y_min)
        new_boxes[:, 2] = minimum(new_boxes[:, 2], ones_like(new_boxes[:, 2]) * x_max)
        new_boxes[:, 3] = minimum(new_boxes[:, 3], ones_like(new_boxes[:, 3]) * y_max)

        new_box_sizes = cls._box_sizes(new_boxes)
        mask = ((new_box_sizes > 0)) & (
            new_box_sizes >= (cls._box_sizes(anns["boxes"]) * min_size)
        )
        return {"boxes": new_boxes[mask], "labels": anns["labels"][mask]}

    def mosaic(self, img: torch.Tensor, target: Dict):
        _, y, x = img.shape
        # find the mosaic centre
        x_centre, y_centre = int(uniform(x / 4, 3 * x / 4)), int(
            uniform(y / 4, 3 * y / 4)
        )
        # the first image will be the top left
        output_annotations = self._trim_annotations(target, 0, x_centre, 0, y_centre)

        for i, (new_img, new_targ) in enumerate(self.random_sample(k=3)):
            if i == 0:
                # bottom left
                img[:, y_centre:, 0:x_centre] = new_img[:, y_centre:, 0:x_centre]
                for key, val in self._trim_annotations(
                    new_targ, 0, x_centre, y_centre, y
                ).items():
                    output_annotations[key] = torch.concat(
                        [output_annotations[key], val]
                    )
            if i == 1:
                # bottom right
                img[:, 0:y_centre, x_centre:] = new_img[:, 0:y_centre, x_centre:]
                for key, val in self._trim_annotations(
                    new_targ, x_centre, x, 0, y_centre
                ).items():
                    output_annotations[key] = torch.concat(
                        [output_annotations[key], val]
                    )
            if i == 2:
                # top right
                img[:, y_centre:, x_centre:] = new_img[:, y_centre:, x_centre:]
                for key, val in self._trim_annotations(
                    new_targ, x_centre, x, y_centre, y
                ).items():
                    output_annotations[key] = torch.concat(
                        [output_annotations[key], val]
                    )
        return img, output_annotations


class DataModule(LightningDataModule):
    def __init__(
        self,
        labelled_training_dataset,
        val_dataset,
        test_dataset,
        num_workers: int = 0,
        unlabelled_training_dataset=None,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.train_ds = ListOfDatasetsWithMosaicing(
            labelled_training_dataset, mosaic=True
        )
        self.val_ds = ListOfDatasetsWithMosaicing(val_dataset)
        self.test_ds = ListOfDatasetsWithMosaicing(test_dataset)

        if unlabelled_training_dataset is not None:
            self.add_unlabelled_training_dataset(unlabelled_training_dataset)

    @classmethod
    def make_dataloader(cls, dataset, num_workers: int, shuffle: bool):
        if hasattr(dataset, "collate_fn"):
            collate_fn = dataset.collate_fn
        else:
            collate_fn = cls.default_collate_fn
        return DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
            # https://github.com/pytorch/pytorch/issues/11201#issuecomment-895047235
            worker_init_fn=set_worker_sharing_strategy,
        )

    def train_dataloader(self):
        return self.make_dataloader(self.train_ds, self.num_workers, shuffle=True)

    def val_dataloader(self):
        return self.make_dataloader(self.val_ds, self.num_workers, shuffle=False)

    def test_dataloader(self):
        return self.make_dataloader(self.test_ds, self.num_workers, shuffle=False)

    @staticmethod
    def default_collate_fn(batch):
        return tuple(zip(*batch))

    def add_unlabelled_training_dataset(self, unlabelled_data):
        new_train_ds = LabelledAndUnlabelledDataset(
            self.train_ds, ListOfDatasetsWithMosaicing(unlabelled_data)
        )
        self.train_ds = new_train_ds
