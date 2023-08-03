import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .base import BaseModel
from .norm import RMSNorm
from .transformer import TransformerBlock


class OcrTransformer(BaseModel):
    kind = "ocr-transformer"

    def __init__(
        self,
        num_chars: int,
        hidden_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        classifier_channels: int = 256,
        dropout_rate: float = 0.2,
        max_len: int = 1024,
    ):
        """
        Args:
            num_chars (int): Number of characters to predict (including the CTC token)
            hidden_size (int): Hidden size for the intermediate results [Default: 256]
            num_layers (int): Number of Transformer layers [Default: 4]
            num_heads (int): Number of attention heads [Default: 8]
            classifier_channels (int): Channels for the classifier layers [Default: 256]
            dropout_rate (float): Dropout probability [Default: 0.2]
            max_len (int): Maximum length for the transformer [Default: 1024]
        """
        super().__init__()
        self.num_chars = num_chars
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.classifier_channels = classifier_channels
        self.dropout_rate = dropout_rate
        self.max_len = max_len

        # An embedding for each character
        self.char_embeddings = nn.Parameter(
            torch.empty(1, self.num_chars, self.hidden_size)
        )
        self.encoder = AutoModel.from_pretrained("WinKawaks/vit-tiny-patch16-224")
        self.encoder.pooler.dense.requires_grad_(False)
        self.encoder_channels = self.encoder.config.hidden_size
        self.encoder_pool = self.encoder.config.patch_size
        self.encoder_to_decoder = (
            nn.Conv2d(
                self.encoder_channels, self.hidden_size, kernel_size=1, bias=False
            )
            if self.encoder_channels != self.hidden_size
            else nn.Identity()
        )
        self.decoder_layers = nn.ModuleList(
            [
                TransformerBlock(
                    self.hidden_size,
                    num_heads=self.num_heads,
                    dropout_rate=self.dropout_rate,
                    max_len=self.max_len,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.classifier_enc = nn.Sequential(
            RMSNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.classifier_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.classifier_channels, self.num_chars),
        )
        self.classifier_dec = nn.Sequential(
            RMSNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.classifier_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.classifier_channels, self.num_chars),
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(self._init_weights)
        nn.init.normal_(self.char_embeddings, mean=0.0, std=0.2)
        # Based on GPT-2, initialise the weight of the layers whose output is added to
        # the residual with a factor 1/sqrt(N) where N is the number of residual layers.
        # In this case there are 2 residual layers per transformer block, one for the
        # attention and the other for the feed forward.
        for name, param in self.named_parameters():
            if name.endswith("proj_out.weight"):
                nn.init.normal_(
                    param, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers)
                )

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            bias = getattr(module, "bias", None)
            if bias is not None:
                nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def config(self) -> Dict:
        return dict(
            num_chars=self.num_chars,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            classifier_channels=self.classifier_channels,
            dropout_rate=self.dropout_rate,
            max_len=self.max_len,
        )

    def forward(
        self, image: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            image (torch.Tensor): Input images
                [Dimension: batch_size x channels x height x width]
            padding_mask (torch.Tensor): Boolean padding mask, where True signifies that
                this part of the image is padding.
                [Dimension: batch_size x height x width]
        """
        batch_size, _, height, width = image.size()
        # The extra +1 on the second dimension is for the cls token.
        # Dimension: batch_size x 1 + (height * width) / patch_size x encoder_channels
        out = self.encoder(image, interpolate_pos_encoding=True).last_hidden_state
        # Restore height and width dimensions and remove the cls token in the beginning,
        # which is irrelevant for our task.
        out = out[:, 1:]
        # From: batch_size x (height * width) / patch_size x encoder_channels
        # To: batch_size x encoder_channels x height / patch_size x width / patch_size
        out = out.transpose(1, 2).reshape(
            batch_size,
            self.encoder_channels,
            height // self.encoder_pool,
            width // self.encoder_pool,
        )
        # Pool the height to 2 to avoid having too many tokens in the decoder.
        # TODO: Find a better way to make it scalable.
        out = F.adaptive_avg_pool2d(out, (2, None))
        out = self.encoder_to_decoder(out)

        # New height and width
        batch_size, _, height, width = out.size()
        # Combine the height and width dimensions into a seq_len.
        # From: batch_size x hidden_size x height x width
        # To: batch_size x seq_len x hidden_size
        # NOTE: height and width are transposed before flattening because only the width
        # is padded in the images (since it uses a fixed height), hence this puts all
        # the padding together at the very end after flattening instead.
        out = out.transpose(-1, -2).flatten(-2).transpose(1, 2)
        if padding_mask is not None:
            # Resize the padding mask to the same size as the features, using
            # nearest neighbour interpolation to keep the values to either 0 or 1.
            # Also combine the height and width dimensions the same way as out
            # NOTE: It's all in one long call, because for some reason mypy thinks that
            # the padding_mask after this can be None, even if it's in the same branch.
            padding_mask = (
                F.interpolate(
                    # A channel dimension needs to be added for the interpolation,
                    # which is removed straight after.
                    # Also needs to be float, but the mask is boolean
                    padding_mask.to(torch.float).unsqueeze(1),
                    size=[height, width],
                    mode="nearest",
                )
                .squeeze(1)
                .to(torch.bool)
                .transpose(-1, -2)
                .flatten(-2)
            )
        # The attention mask needs to be True for all tokens that should be considered
        # in the attention and False for everything else. Since the padding should not
        # be considered it needs to be set to False ,but the padding mask is the
        # inverse, where it is True if that token is padding, hence the negation here.
        # Also need to add a singular dimension for the seq_len_query.
        attn_mask = None if padding_mask is None else ~padding_mask.unsqueeze(1)
        # Expand the query for the whole batch, since the same queries are used.
        query = self.char_embeddings.expand(batch_size, -1, -1)
        for decoder in self.decoder_layers:
            query = decoder(out, query=query, mask=attn_mask)
        out = self.classifier_enc(out)
        query = self.classifier_dec(query)
        # Multiply the encoded images with the characters, creating a character
        # classification for all points.
        # Dimension: batch_size x seq_len x num_chars
        out = torch.matmul(out, query.transpose(1, 2))

        return out, padding_mask
