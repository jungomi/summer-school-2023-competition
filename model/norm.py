import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Norm

    A simplified LayerNorm that only uses the root mean square, i.e. keeps the
    re-scaling property but disposes of the re-centering.
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))

    def extra_repr(self) -> str:
        return f"channels={self.channels}, eps={self.eps}"

    def _norm(self, input: torch.Tensor) -> torch.Tensor:
        return input * torch.rsqrt(
            torch.mean(input**2, dim=-1, keepdim=True) + self.eps
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._norm(input.to(torch.float32)).type_as(input) * self.weight
