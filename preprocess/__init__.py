import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from transformers import TrOCRProcessor

from .image import ImagePreprocessor, greyscale
from .text import TextPreprocessor


@dataclass
class Preprocessor:
    """
    A simple wrapper around the preprocessors to use the TrOCR preprocessing for text
    or a custom character based preprocessor (CTC based), and for the image either the
    TrOCR preprocessing or a custom image preprocessor (keeping the aspect ratio).
    """

    trocr: Optional[TrOCRProcessor] = None
    image_processor: Optional[ImagePreprocessor] = None
    text_processor: Optional[TextPreprocessor] = None
    no_greyscale: bool = False

    def __post_init__(self):
        if self.trocr is None:
            assert (
                self.image_processor is not None and self.text_processor is not None
            ), "trocr=None requires image_processor and text_processor to not be None"

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
        trocr_path = config_path / "preprocessor_config.json"
        img_path = config_path / "image_preprocessor.json"
        text_path = config_path / "text_preprocessor.json"
        config = dict(
            trocr=TrOCRProcessor.from_pretrained(config_path)
            if trocr_path.exists() or not (img_path.exists() and text_path.exists())
            else None,
            image_processor=ImagePreprocessor.from_pretrained(img_path)
            if img_path.exists()
            else None,
            text_processor=TextPreprocessor.from_pretrained(text_path)
            if text_path.exists()
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
        if self.text_processor:
            self.text_processor.save_pretrained(out_path / "text_preprocessor.json")
        if self.trocr:
            self.trocr.save_pretrained(out_path)

    def process_text(self, text: str) -> List[int]:
        if self.text_processor:
            return self.text_processor.encode(text)
        elif self.trocr:
            return self.trocr.tokenizer(text).input_ids
        else:
            raise ValueError("Both trocr and text_processor are None")

    def process_image(self, image: Image.Image) -> torch.Tensor:
        if self.image_processor:
            img_t = TF.to_tensor(image.convert("L"))
            img_t = self.image_processor(img_t)
            img_t = img_t.expand(3, -1, -1)
        elif self.trocr:
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
        else:
            raise ValueError("Both trocr and image_processor are None")
        return img_t

    def decode_text(self, ids: Union[List[int], torch.Tensor]) -> str:
        if self.text_processor:
            return self.text_processor.decode(ids)
        elif self.trocr:
            return self.trocr.tokenizer.decode(ids, skip_special_tokens=True)
        else:
            raise ValueError("Both trocr and text_processor are None")

    def unnormalise_image(self, image: torch.Tensor) -> torch.Tensor:
        if self.image_processor:
            return self.image_processor.unnormalise(image)
        else:
            # Assuming the normalisation produced values in the range [-1, 1], this
            # reverts it to [0, 1].
            return TF.normalize(image, mean=[-1], std=[2])

    def num_tokens(self) -> int:
        if self.text_processor:
            return self.text_processor.num_tokens()
        elif self.trocr:
            return self.trocr.tokenizer.vocab_size
        else:
            raise ValueError("Both trocr and text_processor are None")

    def pad_token_id(self) -> int:
        if self.text_processor:
            return self.text_processor.ctc_id
        elif self.trocr:
            return self.trocr.tokenizer.pad_token_id
        else:
            raise ValueError("Both trocr and text_processor are None")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  trocr={repr(self.trocr)},\n"
            f"  image_processor={self.image_processor},\n"
            f"  text_processor={self.text_processor},\n"
            f"  no_greyscale={self.no_greyscale},\n"
            ")"
        )

    def __call__(self, image: Image.Image, text: str) -> Tuple[torch.Tensor, List[int]]:
        return self.process_image(image), self.process_text(text)
