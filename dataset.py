import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from preprocess import Preprocessor


@dataclass
class SampleInfo:
    path: Path
    text: str


@dataclass
class Sample:
    image: torch.Tensor
    target: List[int]
    text: str
    path: Path


@dataclass
class Batch:
    images: torch.Tensor
    targets: torch.Tensor
    image_padding_masks: torch.Tensor
    target_padding_mask: torch.Tensor
    texts: List[str]
    paths: List[Path]


class Collate:
    """
    A custom collate to pad the images and the text with the corresponding padding
    values.
    """

    def __init__(self, pad_token_id: int, text_min_length: int = 0):
        """
        Args:
            pad_token_id (int): Id of the padding token defined by the tokeniser
            text_min_length (int): Minimum length of the text/target. This can be
                helpful to get a fixed size for hardware optimisations. If the maximum
                length of a text in the batch exceeds this length, the batch will
                e padded to the longest.
                [Default: 0]
        """
        self.pad_token_id = pad_token_id
        self.text_min_length = text_min_length

    def __call__(self, data: List[Sample]) -> Batch:
        images = [d.image for d in data]
        max_width = max([img.size(2) for img in images])
        padded_images = [
            F.pad(img, [0, max_width - img.size(2)], mode="constant", value=1.0)
            for img in images
        ]
        # The padding mask has 1 (True) for pixels that are padding, and 0 (False) for
        # pixels that are not padding.
        image_padding_masks = [
            F.pad(
                torch.zeros((img.size(1), img.size(2)), dtype=torch.long),
                [0, max_width - img.size(2)],
                mode="constant",
                value=1,
            )
            for img in images
        ]
        targets = [d.target for d in data]
        lengths = torch.tensor([len(t) for t in targets])
        max_len = max(int(torch.max(lengths).item()), self.text_min_length)
        # Padding with the pad token, which is ignored by the loss.
        padded_targets = torch.tensor(
            [t + (max_len - len(t)) * [-100] for t in targets],
            dtype=torch.long,
        )
        return Batch(
            images=torch.stack(padded_images, dim=0),
            targets=padded_targets,
            image_padding_masks=torch.stack(image_padding_masks, dim=0),
            target_padding_mask=padded_targets == self.pad_token_id,
            texts=[d.text for d in data],
            paths=[d.path for d in data],
        )


class CompetitionDataset(Dataset):
    def __init__(
        self,
        gt: Union[str, os.PathLike],
        preprocessor: Preprocessor,
        root: Optional[Union[str, os.PathLike]] = None,
        name: Optional[str] = None,
    ):
        self.gt = Path(gt)
        self.root = self.gt.parent if root is None else Path(root)
        self.name = self.gt.stem if name is None else name
        self.preprocessor = preprocessor

        with open(self.gt, "r", encoding="utf-8") as fd:
            reader = csv.reader(
                fd, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=""
            )
            self.data = [
                SampleInfo(path=self.root / line[0], text=line[1]) for line in reader
            ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  gt={repr(self.gt)},\n"
            f"  root={repr(self.root)},\n"
            f"  name={repr(self.name)},\n"
            f"  preprocessor={repr(self.preprocessor)},\n"
            f"  len={len(self)},\n"
            ")"
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Sample:
        sample = self.data[index]
        img = Image.open(sample.path).convert("RGB")
        img_t, target = self.preprocessor(image=img, text=sample.text)

        return Sample(image=img_t, target=target, text=sample.text, path=sample.path)
