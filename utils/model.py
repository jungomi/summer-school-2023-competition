from pathlib import Path

import lavd
import torch.nn as nn
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def save_model(
    logger: lavd.Logger,
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    name: str,
):
    if logger.disabled:
        return
    # path = logger.get_file_path(name)
    path = Path("models") / name
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    processor.save_pretrained(path)


# Unwraps a model to the core model, which can be across multiple layers with
# wrappers such as DistributedDataParallel.
def unwrap_model(model: nn.Module) -> nn.Module:
    while hasattr(model, "module") and isinstance(model.module, nn.Module):
        model = model.module
    return model
