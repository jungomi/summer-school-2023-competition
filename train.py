import argparse
import os
import time
from typing import Dict, List, Optional

import lavd
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.functional.text import char_error_rate, word_error_rate
from torchvision import transforms
from transformers import AutoTokenizer, TrOCRProcessor, VisionEncoderDecoderModel

from dataset import Batch, Collate, CompetitionDataset
from debugger import breakpoint
from ema import AveragedModel
from lr_scheduler import (
    LR_SCHEDULER_KINDS,
    LR_WARMUP_MODES,
    BaseLrScheduler,
    create_lr_scheduler,
)
from preprocess import Preprocessor
from stats import METRICS, METRICS_DICT, average_checkpoint_metric
from stats.log import log_epoch_stats, log_experiment, log_results, log_top_checkpoints
from utils import save_model, split_named_arg, sync_dict_values, unwrap_model

# Revert the normalisation, i.e. going from [-1, 1] to [0, 1]
unnormalise = transforms.Normalize(mean=(-1,), std=(2,))


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
    pretrained = "microsoft/trocr-base-handwritten"

    class lr:
        peak_lr = 2e-5
        scheduler = "inv-sqrt"
        warmup_mode = "linear"
        warmup_steps = 500
        warmup_start_lr = 0.0

    class optim:
        adam_beta2 = 0.98
        adam_eps = 1e-8
        weight_decay = 1e-4
        label_smoothing = 0.1

    class ema:
        decay = 0.9999


def run_train(
    model: VisionEncoderDecoderModel,
    data_loader: DataLoader,
    optimiser: optim.Optimizer,
    epoch: int,
    device: torch.device,
    logger: lavd.Logger,
    lr_scheduler: Optional[BaseLrScheduler],
    amp_scaler: Optional[amp.GradScaler] = None,
    ema_model: Optional[AveragedModel] = None,
) -> Dict:
    torch.set_grad_enabled(True)
    model.train()

    sampler = (
        data_loader.sampler
        if isinstance(data_loader.sampler, DistributedSampler)
        else None
    )
    if sampler is not None:
        sampler.set_epoch(epoch)

    losses = []

    pbar = logger.progress_bar(
        "Train",
        total=len(data_loader.dataset),  # type: ignore
        leave=False,
        dynamic_ncols=True,
    )
    for batch in data_loader:  # type: Batch
        # The last batch may not be a full batch
        curr_batch_size = batch.images.size(0)
        # Automatically run it in mixed precision (FP16) if a scaler is given
        with amp.autocast(enabled=amp_scaler is not None):
            outputs = model(
                pixel_values=batch.images.to(device),
                labels=batch.targets.to(device),
            )
        loss = outputs.loss
        losses.append(loss.item())
        if torch.isnan(loss) or torch.isinf(loss):
            breakpoint("Loss is NaN")
        if lr_scheduler is not None:
            lr_scheduler.adjust_lr()
        optimiser.zero_grad()
        if amp_scaler is None:
            loss.backward()
            # Clip gradients to avoid exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
        else:
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimiser)
            # Clip gradients to avoid exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            amp_scaler.step(optimiser)
            amp_scaler.update()
        if ema_model is not None:
            ema_model.update_parameters(model)

        pbar.update(
            curr_batch_size
            if sampler is None
            else curr_batch_size * sampler.num_replicas
        )
    pbar.close()

    result = dict(loss=torch.mean(torch.tensor(losses)).item())
    # Gather the metrics onto the primary process
    result = sync_dict_values(result, device=device)
    return result


@torch.inference_mode()
def run_validation(
    model: VisionEncoderDecoderModel,
    tokeniser: AutoTokenizer,
    data_loader: DataLoader,
    epoch: int,
    device: torch.device,
    logger: lavd.Logger,
    amp_scaler: Optional[amp.GradScaler] = None,
    ema_model: Optional[AveragedModel] = None,
    name: str = "Validation",
) -> Dict:
    torch.set_grad_enabled(False)
    model.eval()

    sampler = (
        data_loader.sampler
        if isinstance(data_loader.sampler, DistributedSampler)
        else None
    )
    if sampler is not None:
        sampler.set_epoch(epoch)

    cers = []
    wers = []

    pbar = logger.progress_bar(
        name,
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
        cer = char_error_rate(preds=output_text, target=batch.texts).item()
        wer = word_error_rate(preds=output_text, target=batch.texts).item()
        cers.append(cer)
        wers.append(wer)

        pbar.update(
            curr_batch_size
            if sampler is None
            else curr_batch_size * sampler.num_replicas
        )
    pbar.close()

    result = dict(
        cer=torch.mean(torch.tensor(cers)).item(),
        wer=torch.mean(torch.tensor(wers)).item(),
    )
    # Gather the metrics onto the primary process
    result = sync_dict_values(result, device=device)
    result["sample"] = dict(  # type: ignore
        image=unnormalise(batch.images[0].cpu()),
        text=batch.texts[0],
        pred=output_text[0],
    )
    return result


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


def train(
    logger: lavd.Logger,
    model: VisionEncoderDecoderModel,
    optimiser: optim.Optimizer,
    train_data_loader: DataLoader,
    validation_data_loaders: List[DataLoader],
    device: torch.device,
    processor: TrOCRProcessor,
    lr_scheduler: BaseLrScheduler,
    amp_scaler: Optional[amp.GradScaler] = None,
    ema_model: Optional[AveragedModel] = None,
    num_epochs: int = DEFAULTS.num_epochs,
    best_metric: str = METRICS[0].key,
):
    tokeniser = processor
    train_stats: Dict = dict(lr=[], loss=[])
    validation_stats = {
        val_data_loader.dataset.name: dict(cer=[], wer=[])  # type: ignore
        for val_data_loader in validation_data_loaders
    }
    best_checkpoint: Dict = dict(epoch=0)
    for epoch in range(num_epochs):
        actual_epoch = epoch + 1
        epoch_text = "[{current:>{pad}}/{end}] Epoch {epoch}".format(
            current=epoch + 1,
            end=num_epochs,
            epoch=actual_epoch,
            pad=len(str(num_epochs)),
        )
        logger.set_prefix(epoch_text)
        logger.start(epoch_text, prefix=False)
        start_time = time.time()

        logger.start("Train")
        train_result = run_train(
            model,
            train_data_loader,
            optimiser,
            device=device,
            epoch=epoch,
            lr_scheduler=lr_scheduler,
            logger=logger,
            amp_scaler=amp_scaler,
            ema_model=ema_model,
        )
        train_stats["lr"].append(lr_scheduler.lr)
        train_stats["loss"].append(train_result)
        logger.end("Train")

        validation_results = []
        for val_data_loader in validation_data_loaders:
            val_name = val_data_loader.dataset.name  # type: ignore
            val_text = "Validation: {}".format(val_name)
            logger.start(val_text)
            validation_result = run_validation(
                unwrap_model(model if ema_model is None else ema_model),
                tokeniser,
                val_data_loader,
                device=device,
                epoch=epoch,
                logger=logger,
                name=val_text,
                amp_scaler=amp_scaler,
            )
            validation_results.append(dict(name=val_name, **validation_result))
            validation_stats[val_name]["cer"].append(validation_result["cer"])
            validation_stats[val_name]["wer"].append(validation_result["wer"])
            logger.end(val_text)

        with logger.spinner("Checkpoint", placement="right"):
            unwrapped_model = unwrap_model(model if ema_model is None else ema_model)
            best_metric_value = average_checkpoint_metric(
                validation_results,
                key=best_metric,
            )
            if (
                best_checkpoint.get(best_metric) is None
                or best_metric_value < best_checkpoint[best_metric]
            ):
                best_checkpoint = {
                    "epoch": actual_epoch,
                    best_metric: best_metric_value,
                }
            # Only save it as best if the current epoch is the best
            if best_checkpoint["epoch"] == actual_epoch:
                save_model(logger, unwrapped_model, processor, "best")
            save_model(logger, unwrapped_model, processor, "latest")

        with logger.spinner("Logging Data", placement="right"):
            log_results(
                logger,
                actual_epoch,
                dict(lr=lr_scheduler.lr, **train_result),
                validation_results,
                metrics=METRICS,
            )

        with logger.spinner("Best Checkpoints", placement="right"):
            log_top_checkpoints(logger, validation_stats, METRICS)

        time_difference = time.time() - start_time
        epoch_results = [dict(name="Train", **train_result)] + validation_results
        log_epoch_stats(
            logger,
            epoch_results,
            METRICS,
            lr_scheduler=lr_scheduler,
            time_elapsed=time_difference,
        )
        # Report when new best checkpoint was saved
        # Here instead of when saving, to get it after the table of the epoch results.
        if best_checkpoint["epoch"] == actual_epoch:
            logger.println(
                (
                    "{icon:>{pad}} New best checkpoint: "
                    "Epoch {num:0>4} — {metric_name} = {metric_value:.5f} {icon}"
                ),
                icon="🔔",
                pad=logger.indent_size,
                num=best_checkpoint["epoch"],
                metric_name=METRICS_DICT[best_metric].short_name
                or METRICS_DICT[best_metric].name,
                metric_value=best_checkpoint.get(best_metric, "N/A"),
            )
        logger.end(epoch_text, prefix=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt-train",
        dest="gt_train",
        required=True,
        type=str,
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
        default=DEFAULTS.lr.warmup_steps,
        type=int,
        help="Number of linear warmup steps for the learning rate [Default: {}]".format(
            DEFAULTS.lr.warmup_steps
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
        "-p",
        "--pretrained",
        dest="pretrained",
        default=DEFAULTS.pretrained,
        help="Model name or path to the pretrained model [Default: {}]".format(
            DEFAULTS.pretrained
        ),
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
    return parser


def main():
    options = build_parser().parse_args()
    use_cuda = torch.cuda.is_available() and not options.no_cuda
    if use_cuda:
        # Somehow this fixes an unknown error on Windows.
        torch.cuda.current_device()

    if use_cuda and options.num_gpus > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"
        mp.spawn(main_entry, nprocs=options.num_gpus, args=(options, True))
    else:
        main_entry(0, options)


def main_entry(gpu_id: int, options: argparse.Namespace, distributed: bool = False):
    if distributed:
        dist.init_process_group(
            backend="nccl",
            rank=gpu_id,
            world_size=options.num_gpus,
            init_method="env://",
        )
        torch.cuda.set_device(gpu_id)
    torch.manual_seed(options.seed)
    use_cuda = torch.cuda.is_available() and not options.no_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    logger = lavd.Logger(options.name, disabled=gpu_id != 0)
    # Parser needs to be rebuilt, since it can't be serialised and it is needed to even
    # detect the number of GPUs, but it's only used to log it.
    parser = build_parser() if gpu_id == 0 else None

    amp_scaler = amp.GradScaler() if use_cuda and options.fp16 else None
    persistent_workers = not options.no_persistent_workers and options.num_workers > 0

    model = VisionEncoderDecoderModel.from_pretrained(options.pretrained).to(device)
    model.encoder.pooler.requires_grad_(False)
    processor = TrOCRProcessor.from_pretrained(options.pretrained)
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

    spinner = logger.spinner("Loading data", placement="right")
    spinner.start()

    collate = Collate(pad_token_id=tokeniser.pad_token_id)
    train_dataset = CompetitionDataset(
        options.gt_train,
        tokeniser=tokeniser,
        img_preprocessor=img_preprocessor,
        name="Train",
    )
    train_sampler: Optional[DistributedSampler] = (
        DistributedSampler(
            train_dataset, num_replicas=options.num_gpus, rank=gpu_id, shuffle=True
        )
        if distributed
        else None
    )
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=options.batch_size,
        num_workers=options.num_workers,
        # Only shuffle when not using a sampler
        shuffle=train_sampler is None,
        sampler=train_sampler,
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
        validation_sampler: Optional[DistributedSampler] = (
            DistributedSampler(
                validation_dataset,
                num_replicas=options.num_gpus,
                rank=gpu_id,
                shuffle=False,
            )
            if distributed
            else None
        )
        validation_data_loader = DataLoader(
            validation_dataset,
            batch_size=options.batch_size,
            num_workers=options.num_workers,
            shuffle=False,
            sampler=validation_sampler,
            pin_memory=use_cuda,
            # Keep workers alive after the epoch ends to avoid re-initialising them.
            # NOTE: If RAM becomes an issue, set this to false.
            persistent_workers=persistent_workers,
            collate_fn=collate,
        )
        validation_data_loaders.append(validation_data_loader)
    spinner.stop()

    optimiser = optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=options.lr,
        betas=(0.9, options.adam_beta2),
        eps=options.adam_eps,
    )

    if distributed:
        model = DistributedDataParallel(  # type: ignore
            model, device_ids=[gpu_id], find_unused_parameters=False
        )
    ema_model = (
        None
        if options.ema_decay is None
        else AveragedModel(model, ema_alpha=options.ema_decay)
    )

    lr_scheduler = create_lr_scheduler(
        options.lr_scheduler,
        optimiser,
        peak_lr=options.lr,
        warmup_steps=options.lr_warmup,
        warmup_start_lr=options.lr_warmup_start_lr,
        warmup_mode=options.lr_warmup_mode,
        total_steps=len(train_data_loader) * options.num_epochs,
        end_lr=1e-8,
        # To not crash when choosing schedulers that don't support all arguments.
        allow_extra_args=True,
    )

    # Log the details about the experiment
    validation_details = {
        data_loader.dataset.name: {  # type: ignore
            "Size": len(data_loader.dataset),  # type: ignore
        }
        for data_loader in validation_data_loaders
    }
    log_experiment(
        logger,
        train=dict(Size=len(train_dataset)),
        validation=validation_details,
        options=options,
        lr_scheduler=lr_scheduler,
    )
    logger.log_command(parser, options)

    train(
        logger,
        model,
        optimiser,
        train_data_loader,
        validation_data_loaders,
        device=device,
        processor=processor,
        lr_scheduler=lr_scheduler,
        amp_scaler=amp_scaler,
        ema_model=ema_model,
        num_epochs=options.num_epochs,
    )


if __name__ == "__main__":
    main()
