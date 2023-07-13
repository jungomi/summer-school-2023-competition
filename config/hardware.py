from dataclasses import dataclass

import torch
import torch.multiprocessing as mp
from simple_parsing import field


@dataclass
class HardwareConfig:
    """
    Hardware related configuration
    """

    # Random seed for reproducibility
    seed: int = field(default=1234, alias="-s")
    # Batch size
    batch_size: int = field(default=8, alias="-b")
    # Number of workers to use for data loading
    num_workers: int = field(default=mp.cpu_count(), alias="-w")
    # Number of GPUs to use
    num_gpus: int = field(default=torch.cuda.device_count(), alias="-g")
    # Enable mixed precision training (FP16)
    fp16: bool = field(action="store_true")
    # Do not use CUDA even if it's available
    no_cuda: bool = field(action="store_true")
    # Do not persist workers after the epoch ends but reinitialise them at the start of
    # every epoch. (Slower but uses much less RAM)
    no_persistent_workers: bool = field(action="store_true")
