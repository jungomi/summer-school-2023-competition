import os
import tempfile
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Union

import lavd
import psutil
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from config.train import TrainConfig
from dataset import Batch, Collate, OcrDataset  # noqa: F401
from dist import on_main_first
from lr_scheduler import create_lr_scheduler
from model import OcrTransformer, create_model, from_pretrained
from preprocess import ImagePreprocessor, Preprocessor, TextPreprocessor
from stats import METRICS, METRICS_DICT, average_checkpoint_metric
from stats.log import (
    log_config,
    log_epoch_stats,
    log_experiment,
    log_results,
    log_top_checkpoints,
)
from trainer import BaseTrainer, CtcTrainer, HuggingFaceTrainer


def train(
    trainer: BaseTrainer,
    train_data_loader: DataLoader,
    validation_data_loaders: List[DataLoader],
    num_epochs: int = 100,
    best_metric: str = METRICS[0].key,
):
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
        trainer.logger.set_prefix(epoch_text)
        trainer.logger.start(epoch_text, prefix=False)
        start_time = time.time()

        train_result = trainer.train_epoch(train_data_loader, epoch=epoch)
        train_stats["lr"].append(trainer.get_lr())
        train_stats["loss"].append(train_result.loss)

        validation_results = []
        for val_data_loader in validation_data_loaders:
            val_name = val_data_loader.dataset.name  # type: ignore
            validation_result = trainer.validation_epoch(
                val_data_loader,
                epoch=epoch,
                name=val_name,
            )
            validation_results.append(validation_result)
            validation_stats[val_name]["cer"].append(validation_result.cer)
            validation_stats[val_name]["wer"].append(validation_result.wer)

        validation_results_dict = [
            asdict(val_result) for val_result in validation_results
        ]
        with trainer.logger.spinner("Checkpoint", placement="right"):
            best_metric_value = average_checkpoint_metric(
                validation_results_dict,
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
                trainer.save_pretrained("best")
            trainer.save_pretrained("latest")

        with trainer.logger.spinner("Logging Data", placement="right"):
            log_results(
                trainer.logger,
                actual_epoch,
                train_result,
                validation_results,
                metrics=METRICS,
            )

        with trainer.logger.spinner("Best Checkpoints", placement="right"):
            log_top_checkpoints(trainer.logger, validation_stats, METRICS)

        time_difference = time.time() - start_time
        epoch_results = [
            dict(name="Train", **asdict(train_result))
        ] + validation_results_dict
        log_epoch_stats(
            trainer.logger,
            epoch_results,
            METRICS,
            lr_scheduler=trainer.lr_scheduler,
            time_elapsed=time_difference,
        )
        # Report when new best checkpoint was saved
        # Here instead of when saving, to get it after the table of the epoch results.
        if best_checkpoint["epoch"] == actual_epoch:
            trainer.logger.println(
                (
                    "{icon:>{pad}} New best checkpoint: "
                    "Epoch {num:0>4} — {metric_name} = {metric_value:.5f} {icon}"
                ),
                icon="🔔",
                pad=trainer.logger.indent_size,
                num=best_checkpoint["epoch"],
                metric_name=METRICS_DICT[best_metric].short_name
                or METRICS_DICT[best_metric].name,
                metric_value=best_checkpoint.get(best_metric, "N/A"),
            )
        trainer.logger.end(epoch_text, prefix=False)


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

    mmap_dir = None
    if not cfg.hardware.no_mmap:
        tmp_dir = tempfile.TemporaryDirectory()
        mmap_dir = tmp_dir.name

    if num_gpus > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"
        mp.spawn(main_entry, nprocs=num_gpus, args=(True, device_ids, mmap_dir))
    else:
        main_entry(0, device_ids=device_ids, mmap_dir=mmap_dir)


def main_entry(
    rank: int,
    distributed: bool = False,
    device_ids: Optional[List[int]] = None,
    mmap_dir: Optional[Union[str, os.PathLike]] = None,
) -> None:
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
            world_size=cfg.hardware.actual_num_gpus(),
            init_method="env://",
        )
    device_id = rank if device_ids is None else device_ids[rank]
    # The default cuda device is set according to the available GPUs, if they were
    # not limited, it is equivalent to the rank.
    torch.cuda.set_device(device_id)
    torch.manual_seed(cfg.hardware.seed)
    use_cuda = cfg.hardware.use_cuda()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger = lavd.Logger(cfg.name, disabled=rank != 0)

    amp_scaler = amp.GradScaler() if cfg.hardware.use_fp16() else None
    num_workers = cfg.hardware.actual_num_workers()
    persistent_workers = not cfg.hardware.no_persistent_workers and num_workers > 0

    spinner = logger.spinner(f"Loading Model ({device})", placement="right")
    spinner.start()
    if cfg.model.pretrained:
        preprocessor = Preprocessor.from_pretrained(cfg.model.pretrained)
        # With the device as context manager the tensor creations are done onto that
        # device rather than the CPU, which skips the intermediate CPU model that would
        # be caused by Model(...).to(device) before transferring it onto the device.
        # Note: This might not cover all creations, but as long as the best practices
        # are followed, it will work fine. In this particular case it works flawlessly
        # and makes the loading time roughly 4x faster.
        with device:
            model = from_pretrained(cfg.model.pretrained)
    else:
        preprocessor = Preprocessor(
            trocr=TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            if cfg.preprocess.trocr_preprocessing or cfg.preprocess.tokens is None
            else None,
            image_processor=None
            if cfg.preprocess.trocr_preprocessing
            else ImagePreprocessor(
                height=cfg.preprocess.height, no_greyscale=cfg.preprocess.no_greyscale
            ),
            text_processor=None
            if cfg.preprocess.tokens is None
            else TextPreprocessor(tokens=cfg.preprocess.tokens),
        )
        model_kwargs = {}
        if cfg.model.kind == OcrTransformer.kind:
            model_kwargs = dict(
                num_chars=preprocessor.num_tokens(),
                hidden_size=cfg.model.hidden_size,
                num_layers=cfg.model.num_layers,
                num_heads=cfg.model.num_heads,
                classifier_channels=cfg.model.classifier_channels,
                dropout_rate=cfg.model.dropout,
            )
        with device:
            model = create_model(cfg.model.kind, **model_kwargs)
    if isinstance(model, VisionEncoderDecoderModel):
        model.encoder.pooler.requires_grad_(False)
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

    spinner.stop()

    spinner = logger.spinner("Loading data", placement="right")
    spinner.start()

    collate = Collate(
        text_min_length=cfg.text_min_length,
        image_min_width=cfg.image_min_width,
    )
    # Loading the dataset is done on the main process first when using mmap so that
    # only one process needs to preprocess all the data.
    with on_main_first(enabled=mmap_dir is not None):
        train_dataset = OcrDataset(
            cfg.gt_train,
            preprocessor=preprocessor,
            mmap_dir=mmap_dir,
            name="Train",
        )
    train_sampler: Optional[DistributedSampler] = (
        DistributedSampler(
            train_dataset, num_replicas=cfg.hardware.num_gpus, shuffle=True
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
        # Loading the dataset is done on the main process first when using mmap so that
        # only one process needs to preprocess all the data.
        with on_main_first(enabled=mmap_dir is not None):
            validation_dataset = OcrDataset(
                val_gt.path,
                preprocessor=preprocessor,
                mmap_dir=mmap_dir,
                name=val_gt.name,
            )
        validation_sampler: Optional[DistributedSampler] = (
            DistributedSampler(
                validation_dataset,
                num_replicas=cfg.hardware.num_gpus,
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
            model, device_ids=[device_id], find_unused_parameters=False
        )
    if cfg.model.compile:
        # The compilation of the model needs to be after the DDP because there are some
        # additional optimisations for the distributed model.
        model = torch.compile(model, backend=cfg.model.compile)
        if collate.text_min_length == 0:
            logger.eprintln(
                "⚠️ [WARN]: --compile is used with no minimum text length, this will "
                "most likely cause recompilation for each batch. Make sure to "
                "get a fixed size by setting --text-min-length to a value "
                "greater than the potential maximum number of tokens in the targets."
            )
        if preprocessor.image_processor and collate.image_min_width == 0:
            logger.eprintln(
                "⚠️ [WARN]: --compile is used with no minimum image width while using "
                "the height resize preprocessing (not a fixed size), this will "
                "most likely cause recompilation for each batch. Make sure to "
                "get a fixed size by setting --image-min-width to a value "
                "greater than the potential maximum image width."
            )

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

    TrainerClass = HuggingFaceTrainer if cfg.model.kind == "trocr" else CtcTrainer
    trainer = TrainerClass(
        model=model,
        optimiser=optimiser,
        preprocessor=preprocessor,
        device=device,
        logger=logger,
        amp_scaler=amp_scaler,
        lr_scheduler=lr_scheduler,
        ema_alpha=None if cfg.no_ema else cfg.ema_alpha,
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
        preprocessor=preprocessor,
        config=cfg,
    )
    log_config(logger, cfg)
    logger.log_command(parser, options)

    # Disable parallesim for the tokenizers to avoid the warnings as it will be turned
    # of regardless.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    train(
        trainer,
        train_data_loader,
        validation_data_loaders,
        num_epochs=cfg.num_epochs,
    )


if __name__ == "__main__":
    main()
