import torch

from dataset import Batch
from predict import predict_transcription

from .base import BaseTrainer


class HuggingFaceTrainer(BaseTrainer):
    """
    A Trainer for HuggingFace's VisionEncoderDecoderModel. The only difference is the
    input/output of the model and the generation in the predict.
    """

    def forward(self, batch: Batch) -> torch.Tensor:
        # Make the transfer non-blocking, may be slightly faster when used with
        # pin_memory, but since there is no work between this and the forward pass of
        # the model, there might not be any speed up, since it needs to wait anyway.
        # At least it should not hurt.
        images = batch.images.to(self.device, non_blocking=True)
        targets = batch.targets.to(self.device, non_blocking=True)
        target_padding_mask = batch.target_padding_mask.to(
            self.device, non_blocking=True
        )
        outputs = self.model(
            pixel_values=images,
            labels=targets,
            # This is the negation (~) of the padding mask, because the
            # attention mask is 1 for the tokens it needs to attend to, whereas the
            # padding mask is 1 for the padding tokens.
            decoder_attention_mask=~target_padding_mask,
            interpolate_pos_encoding=True,
        )
        return outputs.loss

    def predict(self, batch: Batch) -> str:
        output_text = predict_transcription(
            model=self.unwrap_validation_model(),
            tokeniser=self.preprocessor.trocr.tokenizer,
            batch=batch,
            device=self.device,
            amp_scaler=self.amp_scaler,
        )
        return output_text
