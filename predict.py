import csv
from typing import List, Optional

import lavd
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import VisionEncoderDecoderModel

from config.predict import PredictConfig
from dataset import Batch, Collate, CompetitionDataset
from model import from_pretrained
from preprocess import Preprocessor


@torch.inference_mode()
def predict_transcription(
    model: VisionEncoderDecoderModel,
    preprocessor: Preprocessor,
    batch: Batch,
    device: torch.device,
    amp_scaler: Optional[amp.GradScaler] = None,
) -> List[str]:
    # Automatically run it in mixed precision (FP16) if a scaler is given
    with amp.autocast(enabled=amp_scaler is not None):
        outputs = model.generate(
            pixel_values=batch.images.to(device), interpolate_pos_encoding=True
        )
    output_texts = [preprocessor.decode_text(out) for out in outputs]
    return output_texts


@torch.inference_mode()
def predict_transcription_ctc(
    model: nn.Module,
    preprocessor: Preprocessor,
    batch: Batch,
    device: torch.device,
    amp_scaler: Optional[amp.GradScaler] = None,
    ctc_id: int = 0,
) -> List[str]:
    # Automatically run it in mixed precision (FP16) if a scaler is given
    with amp.autocast(enabled=amp_scaler is not None):
        out, out_padding_mask = model(
            batch.images.to(device), padding_mask=batch.image_padding_mask.to(device)
        )
    output_texts = []
    _, max_ids = torch.max(out, dim=-1)
    for out_ids, out_pad_mask in zip(max_ids, out_padding_mask):
        # Remove padding
        out_ids = out_ids[~out_pad_mask]
        # Get rid of repeating symbols
        out_ids = torch.unique_consecutive(out_ids)
        # Remove CTC tokens
        out_ids = out_ids[out_ids != ctc_id]
        output_texts.append(preprocessor.decode_text(out_ids))
    return output_texts


def main() -> None:
    cfg = PredictConfig.parse_config()
    use_cuda = torch.cuda.is_available() and not cfg.hardware.no_cuda
    if use_cuda:
        # Somehow this fixes an unknown error on Windows.
        torch.cuda.current_device()
    torch.manual_seed(cfg.hardware.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    amp_scaler = amp.GradScaler() if cfg.hardware.use_fp16() else None
    torch.set_grad_enabled(False)

    preprocessor = Preprocessor.from_pretrained(cfg.model)
    # With the device as context manager the tensor creations are done onto that
    # device rather than the CPU, which skips the intermediate CPU model that would
    # be caused by Model(...).to(device) before transferring it onto the device.
    # Note: This might not cover all creations, but as long as the best practices
    # are followed, it will work fine. In this particular case it works flawlessly
    # and makes the loading time roughly 4x faster.
    with device:
        model = from_pretrained(cfg.model)
    if isinstance(model, VisionEncoderDecoderModel):
        # This is a workaround because VisionEncoderDecoderModel.generate falsely
        # rejects the `interpolate_pos_encoding=True`, which needs to be passed to the
        # forward method to use other sizes than the one it was trained on (384x384).
        # It only disables the argument validation, the rest works fine.
        model._validate_model_kwargs = lavd.noop.no_op

        # set special tokens used for creating the decoder_input_ids from the labels
        model.config.decoder_start_token_id = (
            preprocessor.trocr.tokenizer.cls_token_id if preprocessor.trocr else 0
        )
        model.config.pad_token_id = preprocessor.pad_token_id()
        # make sure vocab size is set correctly
        model.config.vocab_size = model.config.decoder.vocab_size

        # set beam search parameters
        model.config.eos_token_id = (
            preprocessor.trocr.tokenizer.sep_token_id if preprocessor.trocr else 0
        )
        model.config.max_new_tokens = 64
        model.config.early_stopping = True
        model.config.no_repeat_ngram_size = 3
        model.config.length_penalty = 2.0
        model.config.num_beams = 4

    input_dir = cfg.file.parent
    out_path = input_dir / "prediction.tsv" if cfg.out is None else cfg.out
    collate = Collate()
    dataset = CompetitionDataset(
        cfg.file,
        preprocessor=preprocessor,
        name=f"Predicting {out_path}",
    )
    data_loader = DataLoader(
        dataset,
        batch_size=cfg.hardware.batch_size,
        num_workers=cfg.hardware.actual_num_workers(),
        shuffle=False,
        pin_memory=use_cuda,
        collate_fn=collate,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_fd = open(out_path, "w", encoding="utf-8")
    writer = csv.writer(out_fd, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar="")

    pbar = tqdm(
        desc=dataset.name,
        total=len(data_loader.dataset),  # type: ignore
        leave=False,
        dynamic_ncols=True,
    )
    for batch in data_loader:  # type: Batch
        # The last batch may not be a full batch
        curr_batch_size = batch.images.size(0)
        if isinstance(model, VisionEncoderDecoderModel):
            output_texts = predict_transcription(
                model=model,
                preprocessor=preprocessor,
                batch=batch,
                device=device,
                amp_scaler=amp_scaler,
            )
        else:
            output_texts = predict_transcription_ctc(
                model=model,
                preprocessor=preprocessor,
                batch=batch,
                device=device,
                amp_scaler=amp_scaler,
            )
        for img_path, output in zip(batch.paths, output_texts):
            writer.writerow([img_path.relative_to(input_dir), output])
        pbar.update(curr_batch_size)
    pbar.close()
    out_fd.close()


if __name__ == "__main__":
    main()
