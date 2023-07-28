from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class TrainResult:
    loss: float
    lr: float


@dataclass
class Sample:
    image: torch.Tensor
    text: str
    pred: str


@dataclass
class ValidationResult:
    name: str
    cer: float
    wer: float
    sample: Optional[Sample] = None
