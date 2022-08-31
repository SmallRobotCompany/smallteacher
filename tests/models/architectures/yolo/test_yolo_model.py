import torch
from smallteacher.models.architectures.yolo import initialize_yolo


def test_image_works():
    model = initialize_yolo(num_classes=2, pretrained=False)
    image = torch.ones((3, 600, 1200))
    # this just checks the image actually passes through the model, as expected
    outputs = model([image, image])
    assert len(outputs) == 2
    assert len(torch.unique(outputs[0]["labels"])) == 2
    target = {
        "boxes": torch.tensor([[1000, 500, 1100, 550]]),
        "labels": torch.tensor([2]).to(torch.int64),
    }
    _ = model([image], [target])
