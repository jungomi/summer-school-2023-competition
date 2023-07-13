from dataclasses import dataclass

from simple_parsing import choice, field

from lr_scheduler import LR_SCHEDULER_KINDS, LR_WARMUP_MODES


@dataclass
class LrConfig:
    """
    Learning rate configuration
    """

    # Peak learning rate to use
    peak_lr: float = field(default=2e-5, alias=["-l", "--learning-rate"])
    # Learning rate scheduler kind to use
    scheduler: str = choice(*LR_SCHEDULER_KINDS, default="inv-sqrt")
    # Number of linear warmup steps for the learning rate
    warmup_steps: int = 500
    # Learning rate to start the warmup from
    warmup_start_lr: float = 0.0
    # How the warmup is performed
    warmup_mode: str = choice(*LR_WARMUP_MODES, default="linear")
