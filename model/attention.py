from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rotary_embedding import RotaryEmbeddingReal


class Attention(nn.Module):
    """
    Attention using the scaled_dot_product_attention and using
    Rotary Positional Embeddings (RoPE).
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

        self.proj_q = nn.Linear(input_size, self.num_heads * self.head_size, bias=False)
        self.proj_k = nn.Linear(input_size, self.num_heads * self.head_size, bias=False)
        self.proj_v = nn.Linear(input_size, self.num_heads * self.head_size, bias=False)
        self.proj_out = nn.Linear(
            self.num_heads * self.head_size, input_size, bias=False
        )
        self.rotary_embedding = RotaryEmbeddingReal(self.head_size, max_len=max_len)
        self.dropout = nn.Dropout(p=dropout_rate)

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
        batch_size, seq_len, _ = input.size()
        if query is None:
            query = input
            seq_len_query = seq_len
        else:
            seq_len_query = query.size(1)
        q = self.proj_q(query).view(
            batch_size, seq_len_query, self.num_heads, self.head_size
        )
        k = self.proj_k(input).view(batch_size, seq_len, self.num_heads, self.head_size)
        v = self.proj_v(input).view(batch_size, seq_len, self.num_heads, self.head_size)

        # Apply rotary positional encoding (RoPE)
        q = self.rotary_embedding(q)
        k = self.rotary_embedding(k)

        # num_heads need to be placed as a batch dimensions, since
        # scaled_dot_product_attention expects the last two dimensions to be sequence
        # length and embedding dimensions, respectively.
        # From: batch_size x seq_len x num_heads x head_size
        # To: batch_size x num_heads x seq_len x head_size
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if mask is not None:
            # Mask needs an extra singular dimensions for the added num_heads dimension
            # in the inputs.
            # From: batch_size x seq_len_query x seq_len
            # To: batch_size x 1 x seq_len_query x seq_len
            mask = mask.unsqueeze(1)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout_rate, is_causal=is_causal
        )
        # Combine the heads and get back to the input size
        # From: batch_size x num_heads x seq_len x head_size
        # To: batch_size x x seq_len x num_heads * head_size
        out = out.transpose(1, 2).reshape(batch_size, seq_len_query, -1)

        out = self.proj_out(out)
        out = self.dropout(out)

        return out
