from typing import Tuple

import torch
import torch.nn as nn


# Computes the complex frequencies
def compute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, step=2, dtype=torch.float) / dim))
    freqs = torch.outer(torch.arange(end), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


class RotaryEmbeddingComplex(nn.Module):
    """
    Rotary Positional Embeddings (RoPE) calculated with complex numbers.

    This may not work for certain hardware configurations, which don't support complex
    number maths, for this use the `RotaryEmbeddingReal` instead.
    """

    freqs_cis: torch.Tensor

    def __init__(self, dim: int, max_len: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.theta = theta
        # The singular dimension is added to make it broadcastable.
        # Dimension: max_len x 1 x dim
        freqs_cis = compute_freqs_cis(dim, max_len, theta=theta).unsqueeze(1)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        *sizes, last = input.size()
        # Last dimension is halved as it is regarded as the tuple of 2 elements that was
        # flattened, e.g. [1, 2, 3, 4] -> [[1, 2], [3, 4]], where each tuple represents
        # a complex number, in this case it would be: 1 + 2i and 3 + 4i
        input_complex = torch.view_as_complex(input.reshape(*sizes, last // 2, 2))
        # Complex multiplication to apply the rotations
        out = input_complex * self.freqs_cis[: input_complex.size(1)]
        # Convert back to real numbers
        out = torch.view_as_real(out).flatten(-2)
        return out


# Computes the real numbered frequencies (cos/sin)
def compute_freqs_cos_sin(
    dim: int, end: int, theta: float = 10000.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, step=2, dtype=torch.float) / dim))
    freqs = torch.outer(torch.arange(end), freqs)
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin


class RotaryEmbeddingReal(nn.Module):
    """
    Rotary Positional Embeddings (RoPE) calculated with real numbers.

    This is the equivalent calculation for the rotary embeddings but with real numbers
    instead of complex numbers, which is supported by all hardware configurations, but
    technically a little slower than the complex calculation if it is supported by the
    hardware.
    """

    freqs_cos: torch.Tensor
    freqs_sin: torch.Tensor

    def __init__(self, dim: int, max_len: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.theta = theta
        # The singular dimension is added to make it broadcastable.
        # Dimension: max_len x 1 x dim
        freqs_cos, freqs_sin = compute_freqs_cos_sin(dim, max_len, theta=theta)
        self.register_buffer("freqs_cos", freqs_cos.unsqueeze(1), persistent=False)
        self.register_buffer("freqs_sin", freqs_sin.unsqueeze(1), persistent=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        *sizes, last = input.size()
        # Last dimension is halved as it is regarded as the tuple of 2 elements that was
        # flattened, e.g. [1, 2, 3, 4] -> [[1, 2], [3, 4]], where each tuple represents
        # a complex number, in this case it would be: 1 + 2i and 3 + 4i
        input_real, input_imaginary = input.reshape(*sizes, last // 2, 2).unbind(-1)
        freqs_cos = self.freqs_cos[: input.size(1)]
        freqs_sin = self.freqs_sin[: input.size(1)]
        out_real = input_real * freqs_cos - input_imaginary * freqs_sin
        out_imaginary = input_real * freqs_sin + input_imaginary * freqs_cos
        # Combine the real and imaginary results and flatten the tuples again
        out = torch.stack([out_real, out_imaginary], dim=-1).flatten(-2)
        return out
