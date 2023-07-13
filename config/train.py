from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from simple_parsing import field

from .hardware import HardwareConfig
from .lr import LrConfig
from .model import ModelConfig
from .optim import OptimConfig
from .preprocess import PreprocessConfig
from .utils import ConfigEntry, NamedPath


@dataclass
class TrainConfig(ConfigEntry):
    """
    Training configuration
    """

    # Path to the ground truth TSV file used for training
    gt_train: Path
    # List of ground truth TSV files used for validation. If no name is specified it
    # uses the name of the ground truth file.
    gt_validation: List[NamedPath] = field(nargs="+", metavar="[NAME=]PATH")
    # Number of epochs to train
    num_epochs: int = field(default=100, alias="-n")
    # Name of the experiment
    name: Optional[str] = None
    # Activate expontential moving average (EMA) of model weights.
    # Optionally, specify the decay / momentum / alpha of the EMA model.
    # The value should be very close to 1, i.e. at least 3-4 9s after the decimal point.
    # If the flag is specified without a value, it defaults to 0.9999.
    ema: Optional[float] = field(default=None, nargs="?", const=0.9999)

    model: ModelConfig = field(default_factory=ModelConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    lr: LrConfig = field(default_factory=LrConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
