from pathlib import Path
from typing import List, Optional

import lavd
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.functional.text import char_error_rate, word_error_rate

from dataset import Batch
from debugger import breakpoint
from dist import sync_dict_values
from ema import AveragedModel
from lr_scheduler import BaseLrScheduler
from model import unwrap_model
from preprocess import Preprocessor

from .result import Sample, TrainResult, ValidationResult
from .utils import set_sampler_epoch


class BaseTrainer:
    """
    A Trainer to handle the training loops and make it easier to extend it with custom
    losses. This is very similar to something like PyTorch Lightning's module, except
    that this is integrated into the Trainer class rather than the Module itself.
    This separates it completely from the Model, so not only can multiple models use the
    same training strategy, but it also means that a Model that is only used for
    inference does not need to define things that are purely for training
    e.g. Lightning requires to create the optimiser from the Module, which needs to know
    the learning rate etc., so if you want to be able to customise it, you would need to
    accept a parameter, which would be useless during inference)

    Furthermore, Lightning abstracts away a lot of things, which caused to also swallow
    up some errors, so I'd rather use the custom code (that I have been using anyway,
    but simply is this style/class) and having some non-zero overhead since it needs to
    cover all possible use cases while including a lot of checks to ensure the users
    utilise it correctly.

    Also this does not do all the multi-processing (DDP) stuff, it that is handled
    separately in the train script, this is really just for the model interactions.
    """

    def __init__(
        self,
        model: nn.Module,
        optimiser: optim.Optimizer,
        device: torch.device,
        preprocessor: Preprocessor,
        logger: lavd.Logger,
        amp_scaler: Optional[amp.GradScaler] = None,
        lr_scheduler: Optional[BaseLrScheduler] = None,
        ema_alpha: Optional[float] = None,
    ):
        self.model = model
        self.optimiser = optimiser
        self.device = torch.device(device)
        self.preprocessor = preprocessor
        self.logger = logger
        self.amp_scaler = amp_scaler
        self.lr_scheduler = lr_scheduler
        self.ema_model = (
            None if ema_alpha is None else AveragedModel(model, ema_alpha=ema_alpha)
        )

    def unwrap_validation_model(self) -> nn.Module:
        return unwrap_model(self.model if self.ema_model is None else self.ema_model)

    def get_lr(self) -> float:
        return (
            self.lr_scheduler.lr
            if self.lr_scheduler
            else self.optimiser.param_groups[0]["lr"]
        )

    def train_epoch(self, data_loader: DataLoader, epoch: int) -> TrainResult:
        torch.set_grad_enabled(True)
        self.model.train()
        num_replicas = set_sampler_epoch(data_loader, epoch=epoch)
        # Zeroing out the gradients here, because during the backward pass the zeroing
        # happens at the end, which saves the memory from it since the
        # zero_grad(set_to_none=True) (default) will eliminate the need to have the
        # gradients in memory, hence resetting them afterwards is beneficial.
        # But for the first step it needs to be done manually.
        self.optimiser.zero_grad()

        self.logger.start("Train")
        losses = []
        pbar = self.logger.progress_bar(
            "Train",
            total=len(data_loader.dataset),  # type: ignore
            leave=False,
            dynamic_ncols=True,
        )
        for batch in data_loader:
            # The last batch may not be a full batch
            curr_batch_size = batch.images.size(0)
            # Automatically run it in mixed precision (FP16) if a scaler is given
            with amp.autocast(enabled=self.amp_scaler is not None):
                loss = self.forward(batch)
            losses.append(loss.item())
            self.backward(loss)
            pbar.update(curr_batch_size * num_replicas)
        pbar.close()

        result = dict(loss=torch.mean(torch.tensor(losses)).item())
        # Gather the metrics onto the primary process
        result = sync_dict_values(result, device=self.device)
        self.logger.end("Train")
        return TrainResult(**result, lr=self.get_lr())

    @torch.inference_mode()
    def validation_epoch(
        self, data_loader: DataLoader, epoch: int, name: Optional[str] = None
    ) -> ValidationResult:
        torch.set_grad_enabled(False)
        self.model.eval()
        num_replicas = set_sampler_epoch(data_loader, epoch=epoch)

        val_text = f"Validation: {name}" if name else "Validation"
        self.logger.start(val_text)
        cers = []
        wers = []
        pbar = self.logger.progress_bar(
            val_text,
            total=len(data_loader.dataset),  # type: ignore
            leave=False,
            dynamic_ncols=True,
        )
        for batch in data_loader:
            # The last batch may not be a full batch
            curr_batch_size = batch.images.size(0)
            # Automatically run it in mixed precision (FP16) if a scaler is given
            with amp.autocast(enabled=self.amp_scaler is not None):
                output_text = self.predict(batch)
            cer = char_error_rate(preds=output_text, target=batch.texts).item()
            wer = word_error_rate(preds=output_text, target=batch.texts).item()
            cers.append(cer)
            wers.append(wer)
            pbar.update(curr_batch_size * num_replicas)
        pbar.close()

        result = dict(
            cer=torch.mean(torch.tensor(cers)).item(),
            wer=torch.mean(torch.tensor(wers)).item(),
        )
        # Gather the metrics onto the primary process
        result = sync_dict_values(result, device=self.device)
        self.logger.end(val_text)
        return ValidationResult(
            **result,
            name=name or "Validation",
            sample=Sample(
                image=self.preprocessor.unnormalise_image(batch.images[0].cpu()),
                text=batch.texts[0],
                pred=output_text[0],
            ),
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        raise NotImplementedError("forward method is not implemented")

    def backward(self, loss: torch.Tensor):
        if torch.isnan(loss) or torch.isinf(loss):
            breakpoint("Loss is NaN")
        if self.lr_scheduler is not None:
            self.lr_scheduler.adjust_lr()
        if self.amp_scaler is None:
            loss.backward()
            # Clip gradients to avoid exploding gradients
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimiser.step()
        else:
            self.amp_scaler.scale(loss).backward()
            self.amp_scaler.unscale_(self.optimiser)
            # Clip gradients to avoid exploding gradients
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.amp_scaler.step(self.optimiser)
            self.amp_scaler.update()
        if self.ema_model is not None:
            self.ema_model.update_parameters(self.model)
        # Zero out the gradients at the end (set_to_none=True by default) to save memory
        # of the gradients since they are now set to None.
        self.optimiser.zero_grad()

    def predict(self, batch: Batch) -> List[str]:
        raise NotImplementedError("predict method is not implemented")

    def save_pretrained(self, name: str) -> Path:
        path = self.logger.get_file_path(name)
        if not self.logger.disabled:
            path.mkdir(parents=True, exist_ok=True)
            model = self.unwrap_validation_model()
            # Unwrapping the module makes the type checking brittle, but this is
            # guaranteed to be any model that implements save_pretrained.
            model.save_pretrained(path, safe_serialization=True)  # type: ignore
            self.preprocessor.save_pretrained(path)
        return path

    def to(self, device: torch.device):
        self.device = device
        self.model.to(device)
        return self
