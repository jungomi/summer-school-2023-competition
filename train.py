import os
import time
from typing import Dict, List, Optional

import lavd
import psutil
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

from config.train import TrainConfig
from dataset import Batch, Collate, CompetitionDataset  # noqa: F401
from debugger import breakpoint
from ema import AveragedModel
from lr_scheduler import BaseLrScheduler, create_lr_scheduler
from predict import predict_transcription
from preprocess import Preprocessor
from stats import METRICS, METRICS_DICT, average_checkpoint_metric
from stats.log import log_epoch_stats, log_experiment, log_results, log_top_checkpoints
from utils import save_model, sync_dict_values, unwrap_model

# Revert the normalisation, i.e. going from [-1, 1] to [0, 1]
unnormalise = transforms.Normalize(mean=(-1,), std=(2,))


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
        # Make the transfer non-blocking, may be slightly faster when used with
        # pin_memory, but since there is no work between this and the forward pass of
        # the model, there might not be any speed up, since it needs to wait anyway.
        # At least it should not hurt.
        images = batch.images.to(device, non_blocking=True)
        targets = batch.targets.to(device, non_blocking=True)
        # The last batch may not be a full batch
        curr_batch_size = batch.images.size(0)
        # Automatically run it in mixed precision (FP16) if a scaler is given
        with amp.autocast(enabled=amp_scaler is not None):
            outputs = model(
                pixel_values=images,
                labels=targets,
                interpolate_pos_encoding=True,
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
    num_epochs: int = 100,
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


def main() -> None:
    # Config needs to be parsed here just to know whether to launch multiple processes.
    cfg = TrainConfig.parse_config()
    num_gpus = cfg.hardware.actual_num_gpus()
    if cfg.hardware.use_cuda():
        # Somehow this fixes an unknown error on Windows.
        torch.cuda.current_device()

    # Limit visible CPUs
    if cfg.hardware.cpus:
        psutil.Process().cpu_affinity(cfg.hardware.cpus.values)
    # Limit visible GPUs
    # Note: Cannot use CUDA_VISIBLE_DEVICES because that needs to be set before the
    # process started. It would technically work for the Multi-GPU case, but not for the
    # Single-GPU that isn't CUDA:0. So this just assigns the GPU ids to the different
    # processes (ranks).
    device_ids = (
        cfg.hardware.gpus.values if cfg.hardware.gpus and num_gpus > 0 else None
    )

    if num_gpus > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"
        mp.spawn(main_entry, nprocs=num_gpus, args=(True, device_ids))
    else:
        main_entry(0, device_ids=device_ids)


def main_entry(
    rank: int, distributed: bool = False, device_ids: Optional[List[int]] = None
):
    # Parser needs to be rebuilt, since it can't be serialised.
    parser = TrainConfig.create_parser()
    # Parsing the args again is necessary because otherwise the arguments are not added
    # properly, it adds them before parsing the arguments.
    # As an alternative, parser._preprocessing() could be called, but might as well be
    # sure that everything is exactly as it would be for the real arguments.
    # So it's easier to just parse them here instead of passing them from the main
    # process.
    options = parser.parse_args()
    cfg: TrainConfig = options.config

    if distributed:
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=cfg.hardware.num_gpus,
            init_method="env://",
        )
    # The default cuda device is set according to the available GPUs, if they were
    # not limited, it is equivalent to the rank.
    torch.cuda.set_device(rank if device_ids is None else device_ids[rank])
    torch.manual_seed(cfg.hardware.seed)
    use_cuda = cfg.hardware.use_cuda()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger = lavd.Logger(cfg.name, disabled=rank != 0)

    amp_scaler = amp.GradScaler() if use_cuda and cfg.hardware.fp16 else None
    num_workers = cfg.hardware.actual_num_workers()
    persistent_workers = not cfg.hardware.no_persistent_workers and num_workers > 0

    spinner = logger.spinner(f"Loading Model ({device})", placement="right")
    spinner.start()
    # With the device as context manager the tensor creations are done onto that device
    # rather than the CPU, which skips the intermediate CPU model that would be caused
    # by Model(...).to(device) before transferring it onto the device.
    # Note: This might not cover all creations, but as long as the best practices are
    # followed, it will work fine. In this particular case it works flawlessly and makes
    # the loading time roughly 4x faster.
    with device:
        model = VisionEncoderDecoderModel.from_pretrained(cfg.model.pretrained)
    model.encoder.pooler.requires_grad_(False)
    processor = TrOCRProcessor.from_pretrained(cfg.model.pretrained)
    tokeniser = processor.tokenizer
    img_preprocessor = (
        processor
        if cfg.preprocess.trocr_preprocessing
        else Preprocessor(
            height=cfg.preprocess.height, no_greyscale=cfg.preprocess.no_greyscale
        )
    )
    spinner.stop()

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
        cfg.gt_train,
        tokeniser=tokeniser,
        img_preprocessor=img_preprocessor,
        no_greyscale=cfg.preprocess.no_greyscale,
        name="Train",
    )
    train_sampler: Optional[DistributedSampler] = (
        DistributedSampler(
            train_dataset, num_replicas=cfg.hardware.num_gpus, rank=rank, shuffle=True
        )
        if distributed
        else None
    )
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=cfg.hardware.batch_size,
        num_workers=num_workers,
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
    for val_gt in cfg.gt_validation:
        validation_dataset = CompetitionDataset(
            val_gt.path,
            tokeniser=tokeniser,
            img_preprocessor=img_preprocessor,
            no_greyscale=cfg.preprocess.no_greyscale,
            name=val_gt.name,
        )
        validation_sampler: Optional[DistributedSampler] = (
            DistributedSampler(
                validation_dataset,
                num_replicas=cfg.hardware.num_gpus,
                rank=rank,
                shuffle=False,
            )
            if distributed
            else None
        )
        validation_data_loader = DataLoader(
            validation_dataset,
            batch_size=cfg.hardware.batch_size,
            num_workers=num_workers,
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
        lr=cfg.lr.peak_lr,
        betas=(0.9, cfg.optim.adam_beta2),
        eps=cfg.optim.adam_eps,
    )

    if distributed:
        model = DistributedDataParallel(  # type: ignore
            model, device_ids=[rank], find_unused_parameters=False
        )
    ema_model = None if cfg.ema is None else AveragedModel(model, ema_alpha=cfg.ema)

    lr_scheduler = create_lr_scheduler(
        cfg.lr.scheduler,
        optimiser,
        peak_lr=cfg.lr.peak_lr,
        warmup_steps=cfg.lr.warmup_steps,
        warmup_start_lr=cfg.lr.warmup_start_lr,
        warmup_mode=cfg.lr.warmup_mode,
        total_steps=len(train_data_loader) * cfg.num_epochs,
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
        num_epochs=cfg.num_epochs,
    )


if __name__ == "__main__":
    main()
