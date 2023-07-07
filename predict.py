import argparse
import csv
from pathlib import Path
from typing import Optional

import torch
import torch.cuda.amp as amp
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, TrOCRProcessor, VisionEncoderDecoderModel

from dataset import Batch, Collate, CompetitionDataset
from preprocess import Preprocessor


@torch.inference_mode()
def predict_transcription(
    model: VisionEncoderDecoderModel,
    tokeniser: AutoTokenizer,
    batch: Batch,
    device: torch.device,
    amp_scaler: Optional[amp.GradScaler] = None,
) -> str:
    # Automatically run it in mixed precision (FP16) if a scaler is given
    with amp.autocast(enabled=amp_scaler is not None):
        outputs = model.generate(pixel_values=batch.images.to(device))
        output_text = tokeniser.batch_decode(outputs, skip_special_tokens=True)
    return output_text


class DEFAULTS:
    seed = 1234
    batch_size = 8
    num_workers = mp.cpu_count()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file",
        metavar="TSV",
        type=Path,
        help="Path to the TSV file to predict",
    )
    parser.add_argument(
        "-o",
        "--out",
        dest="out",
        type=Path,
        help=(
            "Path of the output file "
            "[Default: prediction.tsv in the directory of the input file]"
        ),
    )
    parser.add_argument(
        "-m",
        "--model",
        dest="model",
        required=True,
        help="Model name or path to the pretrained model",
    )
    parser.add_argument(
        "--trocr-preprocessing",
        dest="trocr_preprocessing",
        action="store_true",
        help="Use the default TrOCR prepreocessing instead of the default one",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        dest="batch_size",
        default=DEFAULTS.batch_size,
        type=int,
        help="Size of data batches [Default: {}]".format(DEFAULTS.batch_size),
    )
    parser.add_argument(
        "-w",
        "--workers",
        dest="num_workers",
        default=DEFAULTS.num_workers,
        type=int,
        help="Number of workers for loading the data [Default: {}]".format(
            DEFAULTS.num_workers
        ),
    )
    parser.add_argument(
        "--no-cuda",
        dest="no_cuda",
        action="store_true",
        help="Do not use CUDA even if it's available",
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        action="store_true",
        help="Enable mixed precision training (FP16)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        dest="seed",
        default=DEFAULTS.seed,
        type=int,
        help="Seed for random initialisation [Default: {}]".format(DEFAULTS.seed),
    )
    return parser


def main():
    options = build_parser().parse_args()
    use_cuda = torch.cuda.is_available() and not options.no_cuda
    if use_cuda:
        # Somehow this fixes an unknown error on Windows.
        torch.cuda.current_device()
    torch.manual_seed(options.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    amp_scaler = amp.GradScaler() if use_cuda and options.fp16 else None
    torch.set_grad_enabled(False)

    model = VisionEncoderDecoderModel.from_pretrained(options.model).to(device).eval()
    processor = TrOCRProcessor.from_pretrained(options.model)
    tokeniser = processor.tokenizer
    img_preprocessor = (
        processor
        if options.trocr_preprocessing
        else Preprocessor(height=options.height, no_greyscale=options.no_greyscale)
    )

    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_new_tokens = 64
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    input_dir = options.file.parent
    out_path = input_dir / "prediction.tsv" if options.out is None else options.out
    collate = Collate(pad_token_id=tokeniser.pad_token_id)
    dataset = CompetitionDataset(
        options.file,
        tokeniser=tokeniser,
        img_preprocessor=img_preprocessor,
        name=f"Predicting {out_path}",
    )
    data_loader = DataLoader(
        dataset,
        batch_size=options.batch_size,
        num_workers=options.num_workers,
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
        output_text = predict_transcription(
            model=model,
            tokeniser=tokeniser,
            batch=batch,
            device=device,
            amp_scaler=amp_scaler,
        )
        for img_path, output in zip(batch.paths, output_text):
            writer.writerow([img_path.relative_to(input_dir), output])
        pbar.update(curr_batch_size)
    pbar.close()
    out_fd.close()


if __name__ == "__main__":
    main()
