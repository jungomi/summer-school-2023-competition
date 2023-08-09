from dataclasses import dataclass
from typing import Optional

import pytest
import torch

from trainer.ctc import validate_ctc_inputs

BATCH_SIZE = 4
LAST_INDEX = BATCH_SIZE - 1
NUM_CLASSES = 20


CTC_ERR_HEADER = "CTC loss input validation failed:\n\n"


@dataclass
class Inputs:
    log_probs: torch.Tensor
    targets: torch.Tensor
    input_lengths: torch.Tensor
    target_lengths: torch.Tensor

    @classmethod
    def random(
        cls, seq_len: Optional[int] = None, tar_len: Optional[int] = None
    ) -> "Inputs":
        if seq_len is None:
            seq_len = int(torch.randint(10, 256, (1,)))
        if tar_len is None:
            tar_len = int(torch.randint(1, seq_len // 2, (1,)))
        return cls(
            log_probs=torch.randn((seq_len, BATCH_SIZE, NUM_CLASSES)),
            targets=torch.randint(1, NUM_CLASSES, (BATCH_SIZE, tar_len)),
            input_lengths=torch.arange(seq_len, seq_len - BATCH_SIZE, step=-1),
            target_lengths=torch.arange(tar_len, tar_len - BATCH_SIZE, step=-1),
        )

    def validate(self) -> None:
        validate_ctc_inputs(
            log_probs=self.log_probs,
            targets=self.targets,
            input_lengths=self.input_lengths,
            target_lengths=self.target_lengths,
        )

    # Helper to run the validation with errors
    def expect_validation_error(self, msg: str):
        with pytest.raises(ValueError) as e:
            self.validate()
        assert str(e.value) == CTC_ERR_HEADER + msg


def test_validate_ctc_inputs() -> None:
    Inputs.random().validate()


def test_validate_ctc_inputs_input_lengths_too_large() -> None:
    inputs = Inputs.random()
    seq_len = inputs.log_probs.size(0)
    inputs.input_lengths[0] = seq_len + 1
    inputs.input_lengths[-1] = seq_len + 10

    err_lengths = torch.tensor([seq_len + 1, seq_len + 10])
    err_index = torch.tensor([0, LAST_INDEX])
    inputs.expect_validation_error(
        f"- CTC input lengths too large, must be <={seq_len} but got "
        f"input_lenghts={err_lengths} at index {err_index}"
    )


def test_validate_ctc_inputs_target_lengths_too_large() -> None:
    inputs = Inputs.random()
    tar_len = inputs.targets.size(1)
    inputs.target_lengths[0] = tar_len + 1
    inputs.target_lengths[-1] = tar_len + 10

    err_lengths = torch.tensor([tar_len + 1, tar_len + 10])
    err_index = torch.tensor([0, LAST_INDEX])
    inputs.expect_validation_error(
        f"- CTC target lengths too large, must be <={tar_len} but got "
        f"target_lenghts={err_lengths} at index {err_index}"
    )


def test_validate_ctc_inputs_input_too_small() -> None:
    inputs = Inputs.random()
    tar_len = inputs.targets.size(1)
    # All targets are set to 1 in that sequence, meaning that in order to represent this
    # there needs to be a CTC token between each of them, hence it requires
    # 2 * len - 1 tokens in the input.
    inputs.targets[0] = 1
    inputs.target_lengths[0] = tar_len
    inputs.input_lengths[0] = tar_len

    err_min_lengths = torch.tensor([2 * tar_len - 1])
    err_lengths = torch.tensor([tar_len])
    err_index = torch.tensor([0])
    inputs.expect_validation_error(
        f"- CTC inputs are too small, must be >={err_min_lengths} but got "
        f"input_lenghts={err_lengths} at index {err_index}"
    )


def test_validate_ctc_inputs_targets_ctc_token() -> None:
    inputs = Inputs.random()
    inputs.targets[0, 0] = 0
    inputs.targets[-1, 0] = 0

    err_index = torch.tensor([[0, 0], [LAST_INDEX, 0]])
    inputs.expect_validation_error(
        "- CTC targets must not contain the CTC token (id: 0) but found one at "
        f"index {err_index}"
    )


def test_validate_ctc_inputs_targets_classes_out_of_bounds() -> None:
    inputs = Inputs.random()
    inputs.targets[0, 0] = NUM_CLASSES
    inputs.targets[-1, 0] = NUM_CLASSES + 10

    err_targets = torch.tensor([NUM_CLASSES, NUM_CLASSES + 10])
    err_index = torch.tensor([[0, 0], [LAST_INDEX, 0]])
    inputs.expect_validation_error(
        f"- CTC targets class index out of bounds, must be <={NUM_CLASSES - 1} but "
        f"found targets={err_targets} at index {err_index}"
    )


# This checks all the errors being raised at the same time, since they are aggregated.
# NOTE: As an additional benefit it verifies that incorrect lengths (larger than the
# actual tensor) do not incorrectly affect the verification of the min length.
def test_validate_ctc_inputs_all_errors() -> None:
    inputs = Inputs.random()
    seq_len = inputs.log_probs.size(0)
    tar_len = inputs.targets.size(1)
    inputs.input_lengths[0] = seq_len + 1
    inputs.input_lengths[-1] = seq_len + 10
    inputs.target_lengths[0] = tar_len + 1
    inputs.target_lengths[-1] = tar_len + 10
    inputs.targets[-1, 1] = 0
    inputs.targets[-1, 0] = NUM_CLASSES + 10
    # All targets are set to 1 in that sequence, meaning that in order to represent this
    # there needs to be a CTC token between each of them, hence it requires
    # 2 * len - 1 tokens in the input.
    inputs.targets[1] = 1
    inputs.target_lengths[1] = tar_len + 5
    inputs.input_lengths[1] = tar_len

    err_lengths_inp = torch.tensor([seq_len + 1, seq_len + 10])
    err_lengths_tar = torch.tensor([tar_len + 1, tar_len + 5, tar_len + 10])
    err_lengths_small = torch.tensor([tar_len])
    err_min_lengths = torch.tensor([2 * tar_len - 1])
    err_targets = torch.tensor([NUM_CLASSES + 10])
    err_index = torch.tensor([0, LAST_INDEX])
    err_index_tar = torch.tensor([0, 1, LAST_INDEX])
    err_index_small = torch.tensor([1])
    err_index_ctc = torch.tensor([[LAST_INDEX, 1]])
    err_index_oob = torch.tensor([[LAST_INDEX, 0]])
    inputs.expect_validation_error(
        f"- CTC input lengths too large, must be <={seq_len} but got "
        f"input_lenghts={err_lengths_inp} at index {err_index}\n"
        f"- CTC target lengths too large, must be <={tar_len} but got "
        f"target_lenghts={err_lengths_tar} at index {err_index_tar}\n"
        f"- CTC inputs are too small, must be >={err_min_lengths} but got "
        f"input_lenghts={err_lengths_small} at index {err_index_small}\n"
        "- CTC targets must not contain the CTC token (id: 0) but found one at "
        f"index {err_index_ctc}\n"
        f"- CTC targets class index out of bounds, must be <={NUM_CLASSES - 1} but "
        f"found targets={err_targets} at index {err_index_oob}"
    )
