from pathlib import Path

import lavd
import torch.nn as nn
from transformers import VisionEncoderDecoderModel

from preprocess import Preprocessor


def save_model(
    logger: lavd.Logger,
    model: VisionEncoderDecoderModel,
    processor: Preprocessor,
    name: str,
):
    if logger.disabled:
        return
    path = Path("models") / logger.name / name
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    processor.save_pretrained(path)


# Unwraps a model to the core model, which can be across multiple layers with
# wrappers such as DistributedDataParallel.
def unwrap_model(model: nn.Module) -> nn.Module:
    while hasattr(model, "module") and isinstance(model.module, nn.Module):
        model = model.module
    return model
