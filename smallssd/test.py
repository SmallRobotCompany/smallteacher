import argparse
import sys
import torch
from pathlib import Path

import pytorch_lightning as pl

from smallssd.data import LabelledData, UnlabelledData
from smallssd.config import DATAFOLDER_PATH
from smallssd.keys import CLASSNAME_TO_IDX

from smallteacher.data import DataModule, train_augmentations
from smallteacher.models import FullySupervised, SemiSupervised


def parse_args(args):
    """Parse the arguments."""
    parser = argparse.ArgumentParser(
        description="Testing script for a pytorch lightning model."
    )

    parser.add_argument(
        "--model",
        help="Chooses model architecture",
        type=str,
        default="FRCNN",
        choices=["FRCNN", "RetinaNet", "SSD"],
    )
    parser.add_argument(
        "--version",
        help="lightning_logs version folder of the checkpoint",
        type=str,
        default="version_0",
    )

    return parser.parse_args(args)


def get_checkpoint(version: str) -> Path:
    return list(Path(f"lightning_logs/{version}/checkpoints").glob("best_model*.ckpt"))[
        0
    ]


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    pl.seed_everything(42)

    checkpoint_path = get_checkpoint(args.version)

    try:
        model = FullySupervised.load_from_checkpoint(
            checkpoint_path, model_base=args.model, num_classes=len(CLASSNAME_TO_IDX)
        )
    except RuntimeError:
        model = SemiSupervised.load_from_checkpoint(
            checkpoint_path, model_base=args.model, num_classes=len(CLASSNAME_TO_IDX)
        )

    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
    )

    train_ds, val_ds = LabelledData(root=DATAFOLDER_PATH, eval=False).split(
        transforms=[train_augmentations, None]
    )
    datamodule = DataModule(
        labelled_training_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=LabelledData(root=DATAFOLDER_PATH, eval=True),
        unlabelled_training_dataset=UnlabelledData(root=DATAFOLDER_PATH),
    )

    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
