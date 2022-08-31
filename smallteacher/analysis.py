import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFont

import warnings

from .config import TEST_MAP_KWARGS

from typing import Optional, List, Union, Tuple


def plot_precision_recall_curve(precision_curve: np.ndarray, savedir: Path):
    plt.clf()

    _, ax = plt.subplots(1, 2, figsize=(15, 7))

    for idx, iou_threshold in enumerate(TEST_MAP_KWARGS["iou_thresholds"]):
        wheat_pr_curve = precision_curve[idx, :, 0]
        weed_pr_curve = precision_curve[idx, :, 1]
        ax[0].plot(
            TEST_MAP_KWARGS["rec_thresholds"],
            wheat_pr_curve,
            label=f"IOU: {round(iou_threshold, 2)}",
        )
        ax[1].plot(
            TEST_MAP_KWARGS["rec_thresholds"],
            weed_pr_curve,
            label=f"IOU: {round(iou_threshold, 2)}",
        )

    ax[1].set_xlabel("Recall", fontsize=13)
    ax[0].set_xlabel("Recall", fontsize=13)
    ax[0].set_ylabel("Precision", fontsize=13)

    ax[0].set_title("Crop")
    ax[0].set_title("Wheat")
    ax[1].set_title("Weed")
    ax[1].legend()

    plt.savefig(savedir, bbox_inches="tight", dpi=300)


def _generate_color_palette(num_objects: int):
    palette = np.array([2**25 - 1, 2**15 - 1, 2**21 - 1])
    return [tuple((i * palette) % 255) for i in range(num_objects)]


def draw_bounding_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: Optional[List[str]] = None,
    colors: Optional[
        Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]
    ] = None,
    fill: Optional[bool] = False,
    width: int = 1,
    font: Optional[str] = None,
    font_size: Optional[int] = None,
):

    """
    Draws bounding boxes on given image.
    The values of the input image should be uint8 between 0 and 255.
    If fill is True, Resulting Tensor should be saved as PNG image.
    Args:
        image (Tensor): Tensor of shape (C x H x W) and dtype uint8.
        boxes (Tensor): Tensor of size (N, 4) containing bounding boxes in (xmin, ymin, xmax, ymax) format. Note that
            the boxes are absolute coordinates with respect to the image. In other words: `0 <= xmin < xmax < W` and
            `0 <= ymin < ymax < H`.
        labels (List[str]): List containing the labels of bounding boxes.
        colors (color or list of colors, optional): List containing the colors
            of the boxes or single color for all boxes. The color can be represented as
            PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
            By default, random colors are generated for boxes.
        fill (bool): If `True` fills the bounding box with specified color.
        width (int): Width of bounding box.
        font (str): A filename containing a TrueType font. If the file is not found in this filename, the loader may
            also search in other directories, such as the `fonts/` directory on Windows or `/Library/Fonts/`,
            `/System/Library/Fonts/` and `~/Library/Fonts/` on macOS.
        font_size (int): The requested font size in points.
    Returns:
        img (Tensor[C, H, W]): Image Tensor of dtype uint8 with bounding boxes plotted.
    """

    num_boxes = boxes.shape[0]

    if num_boxes == 0:
        warnings.warn("boxes doesn't contain any box. No box was drawn")
        return image

    if labels is None:
        labels: Union[List[str], List[None]] = [None] * num_boxes  # type: ignore[no-redef]
    elif len(labels) != num_boxes:
        raise ValueError(
            f"Number of boxes ({num_boxes}) and labels ({len(labels)}) mismatch. Please specify labels for each box."
        )

    if colors is None:
        colors = _generate_color_palette(num_boxes)
    elif isinstance(colors, list):
        if len(colors) < num_boxes:
            raise ValueError(
                f"Number of colors ({len(colors)}) is less than number of boxes ({num_boxes}). "
            )
    else:  # colors specifies a single color for all boxes
        colors = [colors] * num_boxes

    colors = [
        (ImageColor.getrgb(color) if isinstance(color, str) else color)
        for color in colors
    ]

    if font is None:
        if font_size is not None:
            warnings.warn(
                "Argument 'font_size' will be ignored since 'font' is not set."
            )
        txt_font = ImageFont.load_default()
    else:
        txt_font = ImageFont.truetype(font=font, size=font_size or 10)

    ndarr = np.transpose(image, [1, 2, 0])
    img_to_draw = Image.fromarray(ndarr)
    img_boxes = boxes.tolist()

    if fill:
        draw = ImageDraw.Draw(img_to_draw, "RGBA")
    else:
        draw = ImageDraw.Draw(img_to_draw)

    for bbox, color, label in zip(img_boxes, colors, labels):  # type: ignore[arg-type]
        if fill:
            fill_color = color + (100,)
            draw.rectangle(bbox, width=width, outline=color, fill=fill_color)
        else:
            draw.rectangle(bbox, width=width, outline=color)

        if label is not None:
            margin = width + 1
            draw.text(
                (bbox[0] + margin, bbox[1] + margin), label, fill=color, font=txt_font
            )

    return img_to_draw
