from dataclasses import dataclass
from typing import Optional

import torch._dynamo
from simple_parsing import choice, field


@dataclass
class ModelConfig:
    """
    Configuration of the model
    """

    # Model name or path to the pretrained model
    pretrained: str = field(default="microsoft/trocr-base-handwritten", alias="-p")
    # Compile the model to optimise its performance. Note: This requires that the inputs
    # are constant, otherwise a recompilation is triggered for each batch. Make sure
    # that the padding mode is used, where each batch is padded to a fix size. The
    # optional argument specifies the backend to be used, which defaults to `inductor`
    compile: Optional[str] = choice(
        *torch._dynamo.list_backends(), default=None, nargs="?", const="inductor"
    )
