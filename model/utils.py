import torch.nn as nn


# Unwraps a model to the core model, which can be across multiple layers with
# wrappers such as DistributedDataParallel.
def unwrap_model(model: nn.Module) -> nn.Module:
    while hasattr(model, "module") and isinstance(model.module, nn.Module):
        model = model.module
    return model
