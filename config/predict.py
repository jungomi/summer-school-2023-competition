from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from simple_parsing import field

from .hardware import HardwareConfig
from .utils import ConfigEntry


@dataclass
class PredictConfig(ConfigEntry):
    """
    Predict configuration
    """

    # Path to the ground truth TSV file to predict
    file: Path = field(positional=True, metavar="TSV")
    # Model name or path to the pretrained model
    model: str = field(alias="-m")
    # Path of the output file
    # [Default: prediction.tsv in the directory of the input file]
    out: Optional[Path] = field(default=None, alias="-o")

    hardware: HardwareConfig = field(default_factory=HardwareConfig)
