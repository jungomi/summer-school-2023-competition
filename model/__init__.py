import json
import os
from pathlib import Path
from typing import Union

import safetensors
from transformers import VisionEncoderDecoderModel

from .base import BaseModel
from .ocr_transformer import OcrTransformer
from .utils import unwrap_model  # noqa: F401

MODEL_KINDS = [OcrTransformer.kind, "trocr"]


def from_pretrained(
    path: Union[str, os.PathLike], **kwargs
) -> Union[BaseModel, VisionEncoderDecoderModel]:
    """
    Creates the model from a pre-trained model.

    Note: This requires a local model.

    Args:
        path (str | os.PathLike): Path to the saved model or the directory
            containing the model, in which case it looks for model.safetensors in
            that directory. To determine which model is used, it will either check the
            metadata in the model checkpoint or look for a metadata.json alongside the
            model.
        **kwargs: Other arguments to pass to the constructor.

    Returns;
        model (BaseModel | VisionEncoderDecoderModel): Model initialised with the
            pre-trained weights and configuration.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "model.safetensors"
    dir = path.parent
    with safetensors.safe_open(path, framework="pt") as f:
        metadata = f.metadata()
    model_kind = metadata.get("kind", None)
    if model_kind is None:
        with open(dir / "metadata.json", "r", encoding="utf-8") as fd:
            metadata = json.load(fd)
            model_kind = metadata["kind"]

    if model_kind == OcrTransformer.kind:
        return OcrTransformer.from_pretrained(path, **kwargs)
    elif model_kind == "trocr":
        return VisionEncoderDecoderModel.from_pretrained(dir, **kwargs)
    else:
        options = " | ".join([repr(k) for k in MODEL_KINDS])
        raise ValueError(
            f"No model available for {model_kind} - must be one of {options}"
        )


def create_model(
    kind: str, *args, **kwargs
) -> Union[BaseModel, VisionEncoderDecoderModel]:
    """
    Creates the model of the given kind.
    Just for convenience.

    Args:
        kind (str): Which kind of model to use
        *args: Arguments to pass to the constructor
        **kwargs: Keyword arguments to pass to the constructor.

    Returns;
        model (BaseModel | VisionEncoderDecoderModel): Model with the config
    """
    if kind == OcrTransformer.kind:
        return OcrTransformer(*args, **kwargs)
    elif kind == "trocr":
        return VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-handwritten", *args, **kwargs
        )
    else:
        options = " | ".join([repr(k) for k in MODEL_KINDS])
        raise ValueError(f"No model available for {kind} - must be one of {options}")
