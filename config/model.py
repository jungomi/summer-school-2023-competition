from dataclasses import dataclass
from typing import Optional

import torch._dynamo
from simple_parsing import choice, field

from model import MODEL_KINDS


@dataclass
class ModelConfig:
    """
    Configuration of the model
    """

    # Model kind to use
    kind: str = choice(default="ocr-transformer", *MODEL_KINDS, alias="-k")
    # Hidden size for the intermediate results
    hidden_size: int = 256
    # Number of Transformer layers
    num_layers: int = 4
    # Number of attention heads
    num_heads: int = 8
    # Channels for the classifier layers
    classifier_channels: int = 256
    # Dropout probability
    dropout: float = 0.2
    # Path to the pretrained model, this overrules most model related parameters and
    # uses the ones from the pretrained model instead. Note: Unlike HunggingFace models,
    # this only works with local models.
    pretrained: Optional[str] = field(default=None, alias="-p")
    # Compile the model to optimise its performance. Note: This requires that the inputs
    # are constant, otherwise a recompilation is triggered for each batch. Make sure
    # that the padding mode is used, where each batch is padded to a fix size. The
    # optional argument specifies the backend to be used, which defaults to `inductor`
    compile: Optional[str] = choice(
        *torch._dynamo.list_backends(), default=None, nargs="?", const="inductor"
    )
