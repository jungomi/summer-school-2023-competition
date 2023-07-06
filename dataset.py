import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer, TrOCRProcessor

from preprocess import Preprocessor


@dataclass
class SampleInfo:
    path: Path
    text: str


@dataclass
class Sample:
    image: torch.Tensor
    target: torch.Tensor
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
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, data: List[Sample]):
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
        max_len = int(torch.max(lengths).item())
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
        tokeniser: AutoTokenizer,
        img_preprocessor: Union[Preprocessor, TrOCRProcessor] = Preprocessor(),
        root: Optional[Union[str, os.PathLike]] = None,
        name: Optional[str] = None,
    ):
        self.gt = Path(gt)
        self.root = self.gt.parent if root is None else Path(root)
        self.name = self.gt.stem if name is None else name
        self.tokeniser = tokeniser
        self.img_preprocessor = img_preprocessor

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
            f"  tokeniser={repr(self.tokeniser)},\n"
            f"  img_preprocessor={repr(self.img_preprocessor)},\n"
            f"  len={len(self)},\n"
            ")"
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Sample:
        sample = self.data[index]
        # Greyscale image
        img = Image.open(sample.path).convert("RGB")
        if self.img_preprocessor:
            if isinstance(self.img_preprocessor, TrOCRProcessor):
                img_t = self.img_preprocessor(img, return_tensors="pt").pixel_values
                img_t = img_t.squeeze(0)
            else:
                img_t = TF.to_tensor(img.convert("L"))
                img_t = self.img_preprocessor(img_t)
                img_t = img_t.expand(3, -1, -1)
        else:
            img_t = TF.to_tensor(img)
        target = self.tokeniser(sample.text).input_ids

        return Sample(image=img_t, target=target, text=sample.text, path=sample.path)
