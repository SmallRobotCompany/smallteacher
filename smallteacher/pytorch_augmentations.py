import torch

from typing import Dict, Optional, Tuple


def bbox_hflip(bbox: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Flip a bounding box horizontally around the y-axis.
    Args:
        bbox (tuple): A bounding box `(x_min, y_min, x_max, y_max)`.
        rows (int): Image rows.
        cols (int): Image cols.
    Returns:
        tuple: A bounding box `(x_min, y_min, x_max, y_max)`.
    """
    with torch.no_grad():
        x_min, y_min, x_max, y_max = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        flipped_box = torch.vstack([cols - x_max, y_min, cols - x_min, y_max]).T
    return flipped_box


def bbox_vflip(bbox: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Flip a bounding box horizontally around the y-axis.
    Args:
        bbox (tuple): A bounding box `(x_min, y_min, x_max, y_max)`.
        rows (int): Image rows.
        cols (int): Image cols.
    Returns:
        tuple: A bounding box `(x_min, y_min, x_max, y_max)`.
    """
    with torch.no_grad():
        x_min, y_min, x_max, y_max = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        flipped_box = torch.vstack([x_min, rows - y_max, x_max, rows - y_min]).T
    return flipped_box


def deterministic_horizontal_flip(
    img: torch.Tensor, targets: Optional[Dict[str, torch.Tensor]]
) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    reverse_index = torch.arange(img.shape[-1] - 1, -1, -1)
    img = img[:, :, reverse_index]
    if targets is not None:
        targets["boxes"] = bbox_hflip(targets["boxes"], img.shape[1], img.shape[2])
    return img, targets


def horizontal_flip(
    img: torch.Tensor, targets: Optional[Dict[str, torch.Tensor]], p: float
) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    if torch.rand(1) < p:
        img, targets = deterministic_horizontal_flip(img, targets)
    return img, targets


def deterministic_vertical_flip(
    img: torch.Tensor, targets: Optional[Dict[str, torch.Tensor]]
) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    reverse_index = torch.arange(img.shape[1] - 1, -1, -1)
    img = img[:, reverse_index, :]
    if targets is not None:
        targets["boxes"] = bbox_vflip(targets["boxes"], img.shape[1], img.shape[2])
    return img, targets


def vertical_flip(
    img: torch.Tensor, targets: Optional[Dict[str, torch.Tensor]], p: float
) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    if torch.rand(1) < p:
        img, targets = deterministic_vertical_flip(img, targets)
    return img, targets
