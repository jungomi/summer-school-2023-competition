import argparse
import datetime
from pathlib import Path

import lightning.pytorch as pl
import torch
import torch.multiprocessing as mp
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.functional import char_error_rate, word_error_rate
from transformers import AutoTokenizer, TrOCRProcessor, VisionEncoderDecoderModel

from dataset import Batch, Collate, CompetitionDataset
from ema import AveragedModel
from lr_scheduler import LR_SCHEDULER_KINDS, LR_WARMUP_MODES
from preprocess import Preprocessor
from utils import split_named_arg


# This is a class containing all defaults, it is not meant to be instantiated, but
# serves as a sort of const struct.
# It uses nested classes, which also don't follow naming conventions because the idea is
# to have it as a sort of struct. This is kind of like having th defaults defined in
# submodules if modules where first-class constructs, but in one file, because these are
# purely for the training and being able to group various options into one category is
# really nice.
# e.g. DEFAULTS.lr.scheduler accesses the default learning rate scheduler, which is in
# the category lr, where there are various other options regarding the learning rate.
class DEFAULTS:
    seed = 1234
    batch_size = 8
    num_workers = mp.cpu_count()
    num_gpus = torch.cuda.device_count()
    num_epochs = 100
    height = 128

    class lr:
        peak_lr = 3e-3
        scheduler = "inv-sqrt"
        warmup_mode = "linear"
        warmup_epochs = 5
        warmup_start_lr = 0.0

    class optim:
        adam_beta2 = 0.98
        adam_eps = 1e-8
        weight_decay = 1e-4
        label_smoothing = 0.1

    class ema:
        decay = 0.9999


class OcrModel(pl.LightningModule):
    def __init__(
        self,
        model: VisionEncoderDecoderModel,
        tokeniser: AutoTokenizer,
        options: argparse.Namespace,
    ):
        super().__init__()
        self.model = model
        self.tokeniser = tokeniser
        self.options = options
        self.ema_model = (
            None
            if options.ema_decay is None
            else AveragedModel(model, ema_alpha=options.ema_decay)
        )
        if self.ema_model:
            for param in self.ema_model.parameters():
                param.requires_grad_(False)

    def training_step(self, batch: Batch, batch_idx):
        outputs = self.model(pixel_values=batch.images, labels=batch.targets)
        loss = outputs.loss

        # Logs metrics for each training_step and add it to the progress bar
        self.log(
            "loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=False
        )
        # and the average across the epoch to the logger
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch: Batch, batch_idx):
        model = self.model if self.ema_model is None else self.ema_model.module
        outputs = model.generate(pixel_values=batch.images)
        output_text = self.tokeniser.batch_decode(outputs, skip_special_tokens=True)
        cer = char_error_rate(preds=output_text, target=batch.texts)
        wer = word_error_rate(preds=output_text, target=batch.texts)

        # Log the average of the metrics across the epoch to the logger
        self.log(
            "cer",
            cer,
            on_step=True,
            on_epoch=False,
            logger=False,
        )
        self.log("wer", wer, on_step=True, on_epoch=False, logger=False)
        self.log(
            "val/cer", cer, on_step=False, on_epoch=True, logger=True, sync_dist=True
        )
        self.log(
            "val/wer", wer, on_step=False, on_epoch=True, logger=True, sync_dist=True
        )

    def predict_step(self, batch: Batch, batch_idx):
        model = self.model if self.ema_model is None else self.ema_model.module
        outputs = model.generate(pixel_values=batch.images)
        output_text = self.tokeniser.batch_decode(outputs, skip_special_tokens=True)
        return output_text

    def configure_optimizers(self):
        optimiser = optim.AdamW(
            self.model.parameters(),
            lr=self.options.lr,
            betas=(0.9, self.options.adam_beta2),
            eps=self.options.adam_eps,
        )
        return optimiser

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema_model:
            self.ema_model.update_parameters(self.model)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt-train",
        dest="gt_train",
        required=True,
        type=Path,
        help="Path to the ground truth JSON file used for training",
    )
    parser.add_argument(
        "--gt-validation",
        dest="gt_validation",
        nargs="+",
        metavar="[NAME=]PATH",
        required=True,
        type=str,
        help=(
            "List of ground truth JSON files used for validation "
            "If no name is specified it uses the name of the ground truth file. "
        ),
    )
    parser.add_argument(
        "--chars",
        dest="chars_file",
        type=Path,
        help="Path to TSV file with the available characters",
    )
    parser.add_argument(
        "--height",
        dest="height",
        default=DEFAULTS.height,
        type=int,
        help="Height the images are resized to [Default: {}]".format(DEFAULTS.height),
    )
    parser.add_argument(
        "-n",
        "--num-epochs",
        dest="num_epochs",
        default=DEFAULTS.num_epochs,
        type=int,
        help="Number of epochs to train [Default: {}]".format(DEFAULTS.num_epochs),
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
        "-g",
        "--gpus",
        dest="num_gpus",
        default=DEFAULTS.num_gpus,
        type=int,
        help="Number of GPUs to use [Default: {}]".format(DEFAULTS.num_gpus),
    )
    parser.add_argument(
        "-l",
        "--learning-rate",
        default=DEFAULTS.lr.peak_lr,
        dest="lr",
        type=float,
        help="Peak learning rate to use [Default: {}]".format(DEFAULTS.lr.peak_lr),
    )
    parser.add_argument(
        "--lr-scheduler",
        dest="lr_scheduler",
        default=DEFAULTS.lr.scheduler,
        choices=LR_SCHEDULER_KINDS,
        help="Learning rate scheduler kind to use [Default: {}]".format(
            DEFAULTS.lr.scheduler
        ),
    )
    parser.add_argument(
        "--lr-warmup",
        dest="lr_warmup",
        default=DEFAULTS.lr.warmup_epochs,
        type=int,
        help="Number of linear warmup steps for the learning rate [Default: {}]".format(
            DEFAULTS.lr.warmup_epochs
        ),
    )
    parser.add_argument(
        "--lr-warmup-start-lr",
        dest="lr_warmup_start_lr",
        default=DEFAULTS.lr.warmup_start_lr,
        type=float,
        help="Learning rate to start the warmup from [Default: {}]".format(
            DEFAULTS.lr.warmup_start_lr
        ),
    )
    parser.add_argument(
        "--lr-warmup-mode",
        dest="lr_warmup_mode",
        default=DEFAULTS.lr.warmup_mode,
        choices=LR_WARMUP_MODES,
        help="How the warmup is performed [Default: {}]".format(
            DEFAULTS.lr.warmup_mode
        ),
    )
    parser.add_argument(
        "--adam-beta2",
        dest="adam_beta2",
        default=DEFAULTS.optim.adam_beta2,
        type=float,
        help="β₂ for the Adam optimiser [Default: {}]".format(
            DEFAULTS.optim.adam_beta2
        ),
    )
    parser.add_argument(
        "--adam-eps",
        dest="adam_eps",
        default=DEFAULTS.optim.adam_eps,
        type=float,
        help="Epsilon for the Adam optimiser [Default: {}]".format(
            DEFAULTS.optim.adam_eps
        ),
    )
    parser.add_argument(
        "--weight-decay",
        dest="weight_decay",
        default=DEFAULTS.optim.weight_decay,
        type=float,
        help="Weight decay of the optimiser [Default: {}]".format(
            DEFAULTS.optim.weight_decay
        ),
    )
    parser.add_argument(
        "--label-smoothing",
        dest="label_smoothing",
        default=DEFAULTS.optim.label_smoothing,
        type=float,
        help="Label smoothing value [Default: {}]".format(
            DEFAULTS.optim.label_smoothing
        ),
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        dest="checkpoint",
        help="Path to the checkpoint to be loaded to resume training",
    )
    parser.add_argument(
        "--trocr-preprocessing",
        dest="trocr_preprocessing",
        action="store_true",
        help="Use the default TrOCR prepreocessing instead of the default one",
    )
    parser.add_argument(
        "--no-greyscale",
        dest="no_greyscale",
        action="store_true",
        help="Do not convert the images to greyscale (keep RGB)",
    )
    parser.add_argument(
        "--no-cuda",
        dest="no_cuda",
        action="store_true",
        help="Do not use CUDA even if it's available",
    )
    parser.add_argument(
        "--no-persistent-workers",
        dest="no_persistent_workers",
        action="store_true",
        help=(
            "Do not persist workers after the epoch ends but reinitialise them at the "
            "start of every epoch. (Slower but uses much less RAM)"
        ),
    )
    parser.add_argument(
        "--name",
        dest="name",
        type=str,
        help="Name of the experiment",
    )
    parser.add_argument(
        "-s",
        "--seed",
        dest="seed",
        default=DEFAULTS.seed,
        type=int,
        help="Seed for random initialisation [Default: {}]".format(DEFAULTS.seed),
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        action="store_true",
        help="Enable mixed precision training (FP16)",
    )
    parser.add_argument(
        "--ema",
        dest="ema_decay",
        type=float,
        # const with nargs=? is essentially a default when the option is specified
        # without an argument (but remains None when it's not supplied).
        const=DEFAULTS.ema.decay,
        nargs="?",
        help=(
            "Activate expontential moving average (EMA) of model weights. "
            "Optionally, specify the decay / momentum / alpha of the EMA model. "
            "The value should be very close to 1, i.e. at least 3-4 9s after "
            "the decimal point. "
            "If the flag is specified without a value, it defaults to {}."
        ).format(DEFAULTS.ema.decay),
    )
    return parser.parse_args()


def main():
    options = parse_args()

    pl.seed_everything(options.seed)
    exp_name = options.name or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    wandb_logger = pl.loggers.WandbLogger(
        name=exp_name,
        project="Summerschool-Competition",
    )
    use_cuda = torch.cuda.is_available() and not options.no_cuda
    persistent_workers = not options.no_persistent_workers and options.num_workers > 0

    model = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-base-handwritten"
    )
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
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

    train_dataset = CompetitionDataset(
        options.gt_train,
        tokeniser=tokeniser,
        img_preprocessor=img_preprocessor,
        name="Train",
    )
    collate = Collate(pad_token_id=train_dataset.tokeniser.pad_token_id)
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=options.batch_size,
        num_workers=options.num_workers,
        shuffle=True,
        pin_memory=use_cuda,
        # Keep workers alive after the epoch ends to avoid re-initialising them.
        # NOTE: If RAM becomes an issue, set this to false.
        persistent_workers=persistent_workers,
        collate_fn=collate,
    )

    validation_data_loaders = []
    for val_gt in options.gt_validation:
        name, gt_path = split_named_arg(val_gt)
        validation_dataset = CompetitionDataset(
            gt_path, tokeniser=tokeniser, img_preprocessor=img_preprocessor, name=name
        )
        validation_data_loader = DataLoader(
            validation_dataset,
            batch_size=options.batch_size,
            num_workers=options.num_workers,
            shuffle=False,
            pin_memory=use_cuda,
            # Keep workers alive after the epoch ends to avoid re-initialising them.
            # NOTE: If RAM becomes an issue, set this to false.
            persistent_workers=persistent_workers,
            collate_fn=collate,
        )
        validation_data_loaders.append(validation_data_loader)

    ocr_model = OcrModel(model, tokeniser=tokeniser, options=options)

    trainer = pl.Trainer(
        max_epochs=options.num_epochs,
        logger=wandb_logger,
        devices=options.num_gpus,
        accelerator="auto" if use_cuda else "cpu",
        precision="16-mixed" if options.fp16 else 32,
        callbacks=[
            # Save the best checkpoint based on the minimum val_cer recorded, and also
            # the last one.
            pl.callbacks.ModelCheckpoint(
                dirpath=f"log/{exp_name}/",
                filename="best",
                save_weights_only=True,
                mode="min",
                monitor="val_cer",
                save_last=True,
                verbose=True,
            )
        ],
        # strategy=pl.strategies.DDPStrategy(find_unused_parameters=False),
    )
    trainer.fit(
        ocr_model,
        train_dataloaders=train_data_loader,
        val_dataloaders=validation_data_loaders,
    )


if __name__ == "__main__":
    main()
