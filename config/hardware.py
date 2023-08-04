from dataclasses import dataclass
from typing import Optional

import torch
import torch.multiprocessing as mp
from simple_parsing import field

from .arg_types.range_list import RangeList


def has_tensor_cores() -> bool:
    """
    Check whether the GPU has Tensor cores and therefore supports FP16.
    This is the case for CUDA compute capability >= 7.0
    """
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major >= 7


@dataclass
class HardwareConfig:
    """
    Hardware related configuration
    """

    # Random seed for reproducibility
    seed: int = field(default=1234, alias="-s")
    # Batch size per GPU
    batch_size: int = field(default=8, alias="-b")
    # Number of workers to use for data loading. If not specified, it will use the
    # number of available CPUs equally distributed across the GPUs.
    # Note: Specifying this value signifies the number of workers per GPU not the total.
    num_workers: Optional[int] = field(default=None, alias="-w", nargs=None)
    # Number of GPUs to use
    num_gpus: int = torch.cuda.device_count()
    # GPUs to use (given as a list or range, similar to taskset). If not specified, will
    # use all available GPUs. Specifying this will overrule the --num-gpus option.
    gpus: Optional[RangeList] = field(default=None, nargs=None)
    # CPUs to use (given as a list or range, similar to taskset). If not specified, will
    # use all available CPUs.
    cpus: Optional[RangeList] = field(default=None, nargs=None)
    # Do not use mixed precision training (FP16) even if it's available.
    no_fp16: bool = field(action="store_true")
    # Do not use CUDA even if it's available
    no_cuda: bool = field(action="store_true")
    # Do not persist workers after the epoch ends but reinitialise them at the start of
    # every epoch. (Slower but uses much less RAM)
    no_persistent_workers: bool = field(action="store_true")
    # Do not memory map the data but keep it in memory/process it on demand.
    no_mmap: bool = field(action="store_true")

    def use_cuda(self) -> bool:
        return torch.cuda.is_available() and not self.no_cuda

    def use_fp16(self) -> bool:
        return self.use_cuda() and has_tensor_cores() and not self.no_fp16

    def actual_num_gpus(self) -> int:
        if self.use_cuda():
            return len(self.gpus.values) if self.gpus else self.num_gpus
        else:
            return 0

    def actual_num_workers(self) -> int:
        num_workers = self.num_workers
        if num_workers is None:
            num_workers = len(self.cpus.values) if self.cpus else mp.cpu_count()
            # When the number of workers was not specified they will be partitioned such
            # that all GPUs get an equal number of workers.
            # This is not done when the option is specified manually, since that is on
            # a per-worker basis rather than the total (similar to batch size)
            if self.actual_num_gpus() > 1:
                num_workers = num_workers // self.actual_num_gpus()
        return num_workers
