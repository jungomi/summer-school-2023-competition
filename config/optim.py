from dataclasses import dataclass


@dataclass
class OptimConfig:
    """
    Configuration of the optimiser
    """

    # β₂ for the Adam optimiser
    adam_beta2: float = 0.98
    # Epsilon for the Adam optimiser
    adam_eps: float = 1e-8
    # Weight decay of the optimiser
    weight_decay: float = 1e-4
    # Label smoothing value
    label_smoothing: float = 0.1
