from dataclasses import dataclass

from simple_parsing import field


@dataclass
class ModelConfig:
    """
    Configuration of the model
    """

    # Model name or path to the pretrained model
    pretrained: str = field(default="microsoft/trocr-base-handwritten", alias="-p")
