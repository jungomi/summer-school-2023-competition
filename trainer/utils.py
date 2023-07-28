from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def set_sampler_epoch(data_loader: DataLoader, epoch: int) -> int:
    """
    Set the sampler to the current epoch and return the number of replicas used by the
    sampler for convenience.

    Args:
        data_loader (DataLoader): DataLoader whose sampler should be set
        epoch (int): Current epoch to set the sampler to

    Returns:
        num_replicas (int): Number of replicas used by the sampler, or 1 if no sampler
            is available.
    """
    sampler = (
        data_loader.sampler
        if isinstance(data_loader.sampler, DistributedSampler)
        else None
    )
    if sampler:
        sampler.set_epoch(epoch)
        return sampler.num_replicas
    else:
        return 1
