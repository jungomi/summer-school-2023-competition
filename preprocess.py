from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchvision import transforms


@dataclass
class Preprocessor:
    height: int = 128
    greyscale_intensity: float = 2.0
    normalise: transforms.Normalize = transforms.Normalize(mean=(0.5,), std=(0.5,))
    no_greyscale: bool = False

    def __call__(self, img: torch.Tensor):
        if not self.no_greyscale:
            img = greyscale(img, intensity=self.greyscale_intensity)
        img = resize(img, new_height=self.height)
        img = self.normalise(img)
        return img


def otsu_threshold(img: torch.Tensor):
    min = int(img.min())
    max = int(img.max())
    num_bins = max - min + 1
    hist = torch.histc(img, bins=num_bins, min=min, max=max)
    bin_centers = torch.arange(min, max + 1)

    weight1 = torch.cumsum(hist, dim=0)
    weight2 = torch.cumsum(hist.flip(dims=(0,)), dim=0)

    hist_centers = hist * bin_centers

    mean1 = torch.cumsum(hist_centers, dim=0) / weight1
    mean2 = (torch.cumsum(hist_centers.flip(dims=(0,)), dim=0) / weight2).flip(
        dims=(0,)
    )

    variance = weight1 * weight2.flip(dims=(0,)) * (mean1 - mean2) ** 2
    threshold = bin_centers[torch.argmax(variance)]
    return threshold


def greyscale(img: torch.Tensor, intensity: float = 2.0) -> torch.Tensor:
    """
    "Binarise" the greyscale image, such that the background is set to white (1) and the
    foreground is intensified, by dividing the values by the given intensity.
    This preserves the nuances in the foreground while the background has been removed.

    Args:
        img (torch.Tensor): Greyscale image with float values in the range [0, 1].
            [Dimension: 1 x height x width]
        intensity (float): Multiplier for the intensity of the foreground.
            [Default: 2.0]

    Returns:
        output (torch.Tensor): The "binarised" image with the foreground intensified
    """
    img_uint = 255 * img
    threshold = otsu_threshold(img_uint)

    bg_pixels = img_uint >= threshold
    output = img.clone()
    output[bg_pixels] = torch.ones_like(output[bg_pixels])
    output[~bg_pixels] /= intensity
    return output


def resize(img: torch.Tensor, new_height: int) -> torch.Tensor:
    """
    Resize a single image to match the new height while keeping its aspect ratio.

    Args:
        img (torch.Tensor): Image to be resized [Dimension: channels x height x width]
        new_height (int): Height the image is resized to.

    Returns:
        img (torch.Tensor): Resized image
    """
    _, height, width = img.size()
    new_width = width * new_height // height
    img = img.unsqueeze(0)
    if new_width >= width:
        img = F.interpolate(
            img, size=(new_height, new_width), mode="bilinear", align_corners=False
        )
    elif new_width < width:
        # For unknown reasons, bilinear gives a worse image quality when downsampling.
        # This is not the case with pillow.
        # The mode "area" is equivalent to adaptive average pool.
        img = F.interpolate(img, size=(new_height, new_width), mode="area")
    img = img.squeeze(0)
    return img
