import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from transformers import TrOCRProcessor

from .image import ImagePreprocessor, greyscale


@dataclass
class Preprocessor:
    """
    A simple wrapper around the preprocessors to use the TrOCR preprocessing for text
    and for the image either the TrOCR preprocessing or a custom image preprocessor
    (keeping the aspect ratio).
    """

    trocr: TrOCRProcessor
    image_processor: Optional[ImagePreprocessor] = None
    no_greyscale: bool = False

    @classmethod
    def from_pretrained(cls, path: Union[str, os.PathLike], **kwargs) -> "Preprocessor":
        """
        Creates the preprocessor from a saved checkpoint

        Args:
            path (str | os.PathLike): Directory with the saved preprocessor configs
            **kwargs: Other arguments to pass to the constructor.

        Returns;
            model (Preprocessor): Preprocessor initialised with the configuration.
        """
        config_path = Path(path)
        img_path = config_path / "image_preprocessor.json"
        config = dict(
            trocr=TrOCRProcessor.from_pretrained(config_path),
            image_processor=ImagePreprocessor.from_pretrained(img_path)
            if img_path.exists()
            else None,
        )
        # Include the manually specified arguments, which allows to overwrite the saved
        # config arguments.
        config.update(kwargs)
        return cls(**config)

    def save_pretrained(self, path: Union[str, os.PathLike]):
        """
        Save the preprocessor config as a JSON file.

        Args:
            path (str | os.PathLike): Directory where the JSON files should be saved.
        """
        out_path = Path(path)
        out_path.mkdir(parents=True, exist_ok=True)
        if self.image_processor:
            self.image_processor.save_pretrained(out_path / "image_preprocessor.json")
        self.trocr.save_pretrained(out_path)

    def process_text(self, text: str) -> List[int]:
        return self.trocr.tokenizer(text).input_ids

    def process_image(self, image: Image.Image) -> torch.Tensor:
        if self.image_processor:
            img_t = TF.to_tensor(image.convert("L"))
            img_t = self.image_processor(img_t)
            img_t = img_t.expand(3, -1, -1)
        else:
            if self.no_greyscale:
                img_t = self.trocr(image, return_tensors="pt").pixel_values
            else:
                # The binarisation (greyscale) takes a Tensor whereas TrOCR
                # preprocessor wants an RGB Image (PIL), so need to convert it back
                # and forth.
                img_grey = TF.to_pil_image(
                    greyscale(TF.to_tensor(image.convert("L")))
                ).convert("RGB")
                img_t = self.trocr(img_grey, return_tensors="pt").pixel_values
            img_t = img_t.squeeze(0)
        return img_t

    def unnormalise_image(self, image: torch.Tensor) -> torch.Tensor:
        if self.image_processor:
            return self.image_processor.unnormalise(image)
        else:
            # Assuming the normalisation produced values in the range [-1, 1], this
            # reverts it to [0, 1].
            return TF.normalize(image, mean=[-1], std=[2])

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  trocr={repr(self.trocr)},\n"
            f"  image_processor={self.image_processor},\n"
            f"  no_greyscale={self.no_greyscale},\n"
            ")"
        )

    def __call__(self, image: Image.Image, text: str) -> Tuple[torch.Tensor, List[int]]:
        return self.process_image(image), self.process_text(text)
