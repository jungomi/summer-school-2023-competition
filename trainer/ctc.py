from typing import List

import torch
import torch.nn.functional as F

from dataset import Batch
from predict import predict_transcription_ctc

from .base import BaseTrainer


def validate_ctc_inputs(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    ctc_id: int = 0,
):
    """
    Validate the CTC inputs because the ctc_loss does not do that and silently gives an
    infinity or incorrect loss, which makes it really hard to know when something goes
    wrong just because the input is invalid, e.g. when the length of the targets is
    bigger than the length of the log_probs.

    Args:
        log_probs (torch.Tensor): Log probabilities for the classes
            [Dimension: seq_len x batch_size x num_classes]
        targets (torch.Tensor): Target classes given as ints, without any CTC tokens.
            [Dimension: batch_size x tar_len]
        input_lengths (torch.Tensor): Lengths of each input (for padded sequences),
            each length must be <=seq_len.
            [Dimension: batch_size]
        target_lengths (torch.Tensor): Lengths of each target (for padded sequences),
            each length must be <=tar_len.
            [Dimension: batch_size]
        ctc_id (int): Id of the CTC token [Default: 0]
    """
    errors = []
    seq_len, batch_size, num_classes = log_probs.size()
    tar_len = targets.size(1)

    inputs_too_large = input_lengths > seq_len
    if torch.any(inputs_too_large):
        errors.append(
            f"CTC input lengths too large, must be <={seq_len} but got "
            f"input_lenghts={input_lengths[inputs_too_large]} at index "
            f"{torch.nonzero(inputs_too_large).squeeze(-1)}"
        )
    targets_too_large = target_lengths > tar_len
    if torch.any(targets_too_large):
        errors.append(
            f"CTC target lengths too large, must be <={tar_len} but got "
            f"target_lenghts={target_lengths[targets_too_large]} at index "
            f"{torch.nonzero(targets_too_large).squeeze(-1)}"
        )

    # How many CTC tokens are needed for the targets, there needs to be at least one
    # CTC token between two consecutive characters if it's the same character.
    min_lengths = torch.tensor(
        [
            # The t[:t_len].size(0) seems redundant, but t_len may be larger than the
            # actual size, so therefore the validation above would fail, but since this
            # will report all errors at once, it needs to be taken into account.
            2 * t[:t_len].size(0) - torch.unique_consecutive(t[:t_len]).size(0)
            for t, t_len in zip(targets, target_lengths)
        ],
        device=input_lengths.device,
    )
    length_too_small = torch.clamp(input_lengths, max=seq_len) < min_lengths
    if torch.any(length_too_small):
        errors.append(
            f"CTC inputs are too small, must be >={min_lengths[length_too_small]} "
            f"but got input_lenghts={input_lengths[length_too_small]} at index "
            f"{torch.nonzero(length_too_small).squeeze(-1)}"
        )

    targets_ctc = targets == ctc_id
    if torch.any(targets_ctc):
        errors.append(
            f"CTC targets must not contain the CTC token (id: {ctc_id}) but found "
            f"one at index {torch.nonzero(targets_ctc).squeeze(-1)}"
        )

    invalid_classes = targets >= num_classes
    if torch.any(invalid_classes):
        errors.append(
            f"CTC targets class index out of bounds, must be "
            f"<={num_classes - 1} but found targets={targets[invalid_classes]} "
            f"at index {torch.nonzero(invalid_classes).squeeze(-1)}"
        )

    if len(errors) > 0:
        err_msg = "\n- ".join(errors)
        raise ValueError(f"CTC loss input validation failed:\n\n- {err_msg}")


class CtcTrainer(BaseTrainer):
    """
    A Trainer for CTC based models.
    """

    def forward(self, batch: Batch) -> torch.Tensor:
        # Make the transfer non-blocking, may be slightly faster when used with
        # pin_memory, but since there is no work between this and the forward pass of
        # the model, there might not be any speed up, since it needs to wait anyway.
        # At least it should not hurt.
        images = batch.images.to(self.device, non_blocking=True)
        targets = batch.targets.to(self.device, non_blocking=True)
        image_padding_mask = batch.image_padding_mask.to(self.device, non_blocking=True)
        target_padding_mask = batch.target_padding_mask.to(
            self.device, non_blocking=True
        )
        out, out_padding_mask = self.model(images, padding_mask=image_padding_mask)
        # CTC expects log probabilities with batch size as second dimension
        # Dimension: seq_len x batch_size x num_chars
        log_probs = torch.log_softmax(out, dim=-1).transpose(0, 1)
        # The lengths are simply the number of non-padding tokens per batch
        input_lengths = torch.sum(~out_padding_mask, dim=-1)
        target_lengths = torch.sum(~target_padding_mask, dim=-1)
        # The ctc inputs are validated because otherwise the loss will just become
        # infinity or negative, making it really hard to debug or know when it goes
        # wrong.
        validate_ctc_inputs(
            log_probs,
            targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
        )
        loss = F.ctc_loss(
            log_probs,
            targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
        )
        return loss

    def predict(self, batch: Batch) -> List[str]:
        output_texts = predict_transcription_ctc(
            model=self.unwrap_validation_model(),
            preprocessor=self.preprocessor,
            batch=batch,
            device=self.device,
            amp_scaler=self.amp_scaler,
        )
        return output_texts
