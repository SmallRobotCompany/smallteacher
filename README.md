# SSOD for Agriculture

This is intended to be a starting point for researching and deploying semi-supervised object detection models for agriculture.

This is achieved by exposing a [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/) module which trains a teacher-student model, given labelled and unlabelled data:

```python
from smallteacher.models import SemiSupervised

model = SemiSupervised(
    model_base="SSD",
    num_classes=2,
)
```
PyTorch torchvision detection models should be drop in replaceable to this pipeline; we currently support Faster R-CNN, Retinanet, YOLO and SSD models.

Given a **Labelled Dataset**, which returns tuples of images and annotations (as expected by any torchvision detection model), and an **Unlabelled Dataset** (which returns only unlabelled images), users can construct a DataModule which can be used to train this model:

```python
from smallteacher.data import DataModule

datamodule = DataModule(
    labelled_train_ds,
    labelled_val_ds,
    labelled_test_ds
)
datamodule.add_unlabelled_data(unlabelled_ds)
```

An example of this code being applied to a [semi-supervised dataset](https://github.com/SmallRobotCompany/smallssd) is available in the [`smallSSD`](smallSSD) folder.

### Installation

`smallteacher` can be installed with the following command:

```bash
pip install smallteacher
```

### License
`smallteacher` has a [Creative Commons Attribution-NonCommercial 4.0 International](https://github.com/smallrobotcompany/smallteacher/blob/main/LICENSE) license.
