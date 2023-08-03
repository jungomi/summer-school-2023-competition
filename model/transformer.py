from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention
from .norm import RMSNorm


class FeedForward(nn.Module):
    """
    A feed forward module using SwiGLU (Swish Gated Linear Unit) as the activation
    function.
    """

    def __init__(
        self, input_size: int, hidden_size: int = 256, dropout_rate: float = 0.0
    ):
        super().__init__()
        self.proj_in = nn.Linear(input_size, hidden_size, bias=False)
        self.proj_gate = nn.Linear(input_size, hidden_size, bias=False)
        self.proj_out = nn.Linear(hidden_size, input_size, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = F.silu(self.proj_in(input)) * self.proj_gate(input)
        out = self.proj_out(out)
        out = self.dropout(out)
        return out


class TransformerBlock(nn.Module):
    """
    A Transformer block with a (Multi-Head)Attention block and a feed forward.

    Includes improvements that have been integrated recently, this is essentially what
    is used in models like LLaMa.

    Notable changes:
        - No bias in any Linear
        - Rotary Positional Embedding (RoPE)
        - Pre-Norm
        - Root Mean Square Norm (RMSNorm)
    """

    def __init__(
        self,
        input_size: int = 256,
        num_heads: int = 8,
        dropout_rate: float = 0.0,
        max_len: int = 1024,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_heads = num_heads
        self.head_size = input_size // num_heads
        self.dropout_rate = dropout_rate
        self.max_len = max_len

        self.attention = Attention(
            input_size, num_heads=num_heads, dropout_rate=dropout_rate, max_len=max_len
        )
        self.feed_forward = FeedForward(input_size, hidden_size=4 * input_size)
        self.norm_attn = RMSNorm(input_size)
        self.norm_ff = RMSNorm(input_size)

    def forward(
        self,
        input: torch.Tensor,
        query: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): Input of the attention, which is always used to
                calculate key and value, but also the query unless it is given
                separately.
                [Dimension: batch_size x seq_len x input_size]
            query (torch.Tensor, optional): Calculate the query from this. If not given,
                the query will be calculated from the input (self-attention).
                [Dimension: batch_size x seq_len_query x input_size]
            mask (torch.Tensor, optional): Attention mask to decide which tokens need to
                be attended to.
                See torch.nn.functional.scaled_dot_product_attention for details.
                But usually it is expected to be a boolean mask where True indicates the
                token should be considered for the attention as a matrix between the
                query and key/value tokens, hence the last two dimensions need to be the
                seq_len of the query and the input respectively. If no query is given,
                it will be the same as the input.
                [Dimension: batch_size x seq_len(_query) x seq_len]
            is_causal (bool): Whether it's a causal attention, i.e. can only attend to
                tokens before the current one.
                [Default: False]
        """
        # The residual is added to the query, but if no query is given the input will be
        # used as a query, hence the residual needs to do the same.
        out = input if query is None else query
        out = out + self.attention(
            self.norm_attn(input), query=query, mask=mask, is_causal=is_causal
        )
        out = out + self.feed_forward(self.norm_ff(out))
        return out
