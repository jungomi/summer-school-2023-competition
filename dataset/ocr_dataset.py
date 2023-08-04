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

from .mmap import MmapReader, MmapWriter


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
    image_padding_mask: torch.Tensor
    target_padding_mask: torch.Tensor
    texts: List[str]
    paths: List[Path]


class Collate:
    """
    A custom collate to pad the images and the text with the corresponding padding
    values.
    """

    def __init__(
        self,
        image_pad_value: float = 1.0,
        text_pad_value: int = -100,
        text_min_length: int = 0,
        image_min_width: int = 0,
    ):
        """
        Args:
            image_pad_value (float): Value to pad the images with.
                [Default: 1.0, which is white even after normalisation]
            text_pad_value (int): Value to pad the text/target with.
                [Default: -100 which is automatically ignored in the cross entropy loss]
            text_min_length (int): Minimum length of the text/target. This can be
                helpful to get a fixed size for hardware optimisations. If the maximum
                length of a text in the batch exceeds this length, the batch will
                be padded to the longest.
                [Default: 0]
            image_min_width (int): Minimum width of the image. This can be
                helpful to get a fixed size for hardware optimisations. If the maximum
                width of an image in the batch exceeds this length, the batch will
                be padded to the longest.
                [Default: 0]
        """
        self.image_pad_value = image_pad_value
        self.text_pad_value = text_pad_value
        self.text_min_length = text_min_length
        self.image_min_width = image_min_width

    def __call__(self, data: List[Sample]) -> Batch:
        images = [d.image for d in data]
        max_width = max(max([img.size(2) for img in images]), self.image_min_width)
        padded_images = [
            F.pad(
                img,
                [0, max_width - img.size(2)],
                mode="constant",
                value=self.image_pad_value,
            )
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
            [t + (max_len - len(t)) * [self.text_pad_value] for t in targets],
            dtype=torch.long,
        )
        return Batch(
            images=torch.stack(padded_images, dim=0),
            targets=padded_targets,
            image_padding_mask=torch.stack(image_padding_masks, dim=0),
            target_padding_mask=padded_targets == self.text_pad_value,
            texts=[d.text for d in data],
            paths=[d.path for d in data],
        )


class OcrDataset(Dataset):
    def __init__(
        self,
        gt: Union[str, os.PathLike],
        preprocessor: Preprocessor,
        root: Optional[Union[str, os.PathLike]] = None,
        mmap_dir: Optional[Union[str, os.PathLike]] = None,
        name: Optional[str] = None,
    ):
        """
        Args:
            gt (str | os.PathLike): Path to ground truth TSV file.
            preprocessor (Preprocessor): Preprocessor with a text and image processor.
            root (str | os.PathLike, optional): Path to the root of the images.
                [Default: Directory of the ground truth TSV file, i.e. all paths are
                relative to the ground truth file]
            mmap_dir (str | os.PathLike, optional): Base directory for the memory
                mapping. A subdirectory will be created with the name of the dataset in
                order to be able to have multiple datasets with memory mappings without
                having to manually specify a different directory for each.
                If not specified, no memory mapping is used.
            name (str, optional): Name of the dataset
                [Default: Name of the ground truth file and its parent directory]
        """
        self.gt = Path(gt)
        self.root = self.gt.parent if root is None else Path(root)
        self.name = f"{self.gt.parent.name}/{self.gt.stem}" if name is None else name
        self.preprocessor = preprocessor

        with open(self.gt, "r", encoding="utf-8") as fd:
            reader = csv.reader(
                fd, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=""
            )
            self.data_info = [
                SampleInfo(path=self.root / line[0], text=line[1]) for line in reader
            ]

        self.mmap_reader = None
        if mmap_dir is not None:
            # Create a subdirectory with the name of the dataset to not conflict with
            # others.
            mmap_dir = Path(mmap_dir) / self.name
            if not MmapReader.has_mmap(mmap_dir):
                # mmap doesn't exist yet, so load the data and create the mmap files
                # Notably, this preprocesses the text, which means that it is only done
                # a single time instead of every time it is loaded, making it much
                # faster to load.
                # TODO: Check whether preprocessing the image gives any speed up, which
                # could be the case since the resizing can reduce it dramatically, but
                # that should probably stored separately rather than just serialising
                # it.
                mmap_writer = MmapWriter(mmap_dir)
                for sample in self.data_info:
                    encoded_text = self.preprocessor.process_text(sample.text)
                    mmap_writer.write(encoded_text)
                del mmap_writer

            # Initialise the mmap reader so it can be accessed to retrieve the data
            # through it.
            self.mmap_reader = MmapReader.open(mmap_dir)

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
        return len(self.mmap_reader) if self.mmap_reader else len(self.data_info)

    def __getitem__(self, index: int) -> Sample:
        sample = self.data_info[index]
        img = Image.open(sample.path).convert("RGB")
        img_t = self.preprocessor.process_image(img)
        if self.mmap_reader:
            target = self.mmap_reader[index]
        else:
            target = self.preprocessor.process_text(sample.text)
        return Sample(image=img_t, target=target, text=sample.text, path=sample.path)
