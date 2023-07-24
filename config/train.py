from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from simple_parsing import field

from .arg_types import NamedPath
from .hardware import HardwareConfig
from .lr import LrConfig
from .model import ModelConfig
from .optim import OptimConfig
from .preprocess import PreprocessConfig
from .utils import ConfigEntry


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
    # α / momentum / decay of the EMA model.
    # The value should be very close to 1, i.e. at least 3-4 9s after the decimal point.
    ema_alpha: float = 0.9999
    # Do not use the exponential moving average (EMA) of model weights.
    no_ema: bool = field(action="store_true")
    # Minimum length of the text/target. Set this to something higher than the potential
    # maximum text length (in tokens) to get a fixed size of inputs, which can be useful
    # for hardware optimisations.
    text_min_length: int = 0
    # Minimum width of the image. Set this to something higher than the potential
    # maximum image width to get a fixed size of inputs when using the height resize
    # rather than the TrOCR fixed size preprocessing, which can be useful
    # for hardware optimisations.
    image_min_width: int = 0

    model: ModelConfig = field(default_factory=ModelConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    lr: LrConfig = field(default_factory=LrConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
