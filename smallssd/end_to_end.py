"""
Run the model end to end
"""
import argparse
import sys
import torch
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from smallteacher.data import DataModule, train_augmentations
from smallteacher.models import FullySupervised, SemiSupervised
from smallteacher.constants import Metrics
from smallteacher.config import BEST_MODEL_NAME

from smallssd.data import LabelledData, UnlabelledData
from smallssd.config import DATAFOLDER_PATH
from smallssd.keys import CLASSNAME_TO_IDX

import mlflow
import mlflow.pytorch


def parse_args(args):
    """Parse the arguments."""
    parser = argparse.ArgumentParser(
        description="Simple training script for training a pytorch lightning model."
    )

    parser.add_argument(
        "--model",
        help="Chooses model architecture",
        type=str,
        default="FRCNN",
        choices=["FRCNN", "RetinaNet", "SSD"],
    )
    parser.add_argument(
        "--workers", help="Number of dataloader workers", type=int, default="1"
    )

    parser.add_argument(
        "--mlflow_experiment", type=str, default="pytorch_lightning_experiment"
    )
    parser.add_argument("--seed", type=int, default="42")

    return parser.parse_args(args)


def get_checkpoint(version: int) -> Path:
    return list(
        Path(f"lightning_logs/version_{version}/checkpoints").glob("best_model*.ckpt")
    )[0]


def train_fully_supervised(datamodule, model_name) -> int:
    model = FullySupervised(
        model_base=model_name,
        num_classes=len(CLASSNAME_TO_IDX),
    )
    fully_supervised_trainer = pl.Trainer(
        callbacks=[
            EarlyStopping(monitor=Metrics.MAP, mode="max", patience=10),
            ModelCheckpoint(filename=BEST_MODEL_NAME, monitor=Metrics.MAP, mode="max"),
        ],
        gpus=torch.cuda.device_count(),
    )
    fully_supervised_trainer.fit(model, datamodule=datamodule)

    best_model = FullySupervised.load_from_checkpoint(
        get_checkpoint(fully_supervised_trainer.logger.version),
        model_base=model_name,
        num_classes=len(CLASSNAME_TO_IDX),
    )
    fully_supervised_trainer.test(best_model, datamodule=datamodule)

    return fully_supervised_trainer.logger.version


def train_teacher_student(datamodule, model_name, model_checkpoint) -> int:
    unlabelled_ds = UnlabelledData(root=DATAFOLDER_PATH)
    datamodule.add_unlabelled_training_dataset(unlabelled_ds)

    org_model = FullySupervised.load_from_checkpoint(
        model_checkpoint,
        model_base=model_name,
        num_classes=len(CLASSNAME_TO_IDX),
    )
    model = SemiSupervised(
        trained_model=org_model.model,
        model_base=model_name,
        num_classes=len(CLASSNAME_TO_IDX),
    )

    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        callbacks=[
            EarlyStopping(monitor=Metrics.MAP, mode="max", patience=10),
            ModelCheckpoint(filename=BEST_MODEL_NAME, monitor=Metrics.MAP, mode="max"),
        ],
    )

    trainer.fit(model, datamodule=datamodule)
    best_model = SemiSupervised.load_from_checkpoint(
        get_checkpoint(trainer.logger.version),
        model_base=model_name,
        num_classes=len(CLASSNAME_TO_IDX),
    )
    trainer.test(best_model, datamodule=datamodule)
    return best_model


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    mlflow.set_experiment(experiment_name=args.mlflow_experiment)
    pl.seed_everything(args.seed)

    datamodule = DataModule(
        *LabelledData(root=DATAFOLDER_PATH, eval=False).split(
            transforms=[train_augmentations, None]
        ),
        test_dataset=LabelledData(root=DATAFOLDER_PATH, eval=True),
        num_workers=args.workers,
    )

    mlflow.pytorch.autolog()
    with mlflow.start_run(run_name=f"{args.model}_fully_supervised"):
        version_id = train_fully_supervised(datamodule, args.model)
        best_model_checkpoint = get_checkpoint(version_id)
    with mlflow.start_run(run_name=f"{args.model}_teacher_student"):
        best_model = train_teacher_student(
            datamodule, args.model, best_model_checkpoint
        )
        mlflow.pytorch.log_model(best_model.model, artifact_path="model")


if __name__ == "__main__":
    main()
