import argparse
import time
from typing import Dict, List, Optional

import lavd
import torch

from config.utils import ConfigEntry
from lr_scheduler import BaseLrScheduler

from .dict_utils import get_recursive
from .metric import Metric


def log_top_checkpoints(
    logger: lavd.Logger,
    results: Dict,
    metrics: List[Metric],
    k: int = 5,
):
    total: Dict[str, Dict] = {}
    for name, result in results.items():
        lines = []
        lines.append("# Best Checkpoints - {}".format(name))
        for metric in metrics:
            val = get_recursive(result, key=metric.key)
            values = torch.tensor(val)
            descending = metric.order == "max"
            title = "## {}".format(metric.name)
            if title in total:
                total[title]["values"].append(values)
            else:
                total[title] = dict(values=[values], descending=descending)
            lines.append("")
            lines.append(title)
            lines.append("")
            sorted_metric = torch.sort(values, descending=descending)
            for i, (value, index) in enumerate(zip(*sorted_metric)):
                if i >= k:
                    break
                lines.append(
                    ("{i}. {path} - {value:.5f}").format(
                        i=i + 1,
                        path=logger.get_file_path(
                            "model", step=index + 1, extension=".pt"
                        ).parent.as_posix(),
                        value=value.item(),
                    )
                )
        markdown = "\n".join(lines)
        logger.log_markdown(markdown, "best/{}".format(name))

    lines = []
    lines.append("# Best Checkpoints - Average")
    for title, result in total.items():
        lines.append("")
        lines.append(title)
        lines.append("")
        # Take the mean of all validation results
        values = torch.mean(torch.stack(result["values"], dim=0), dim=0)
        sorted_metric = torch.sort(values, descending=result["descending"])
        for i, (value, index) in enumerate(zip(*sorted_metric)):
            if i >= k:
                break
            lines.append(
                ("{i}. {path} - {value:.5f}").format(
                    i=i + 1,
                    path=logger.get_file_path(
                        "model", step=index + 1, extension=".pt"
                    ).parent.as_posix(),
                    value=value.item(),
                )
            )
    markdown = "\n".join(lines)
    logger.log_markdown(markdown, "best")


def log_experiment(
    logger: lavd.Logger,
    train: Dict,
    validation: Dict,
    options: argparse.Namespace,
    lr_scheduler: BaseLrScheduler,
    config: ConfigEntry,
):
    infos = {
        "Train Dataset": train,
        "Validation Dataset": validation,
    }
    sections = {
        "LR Scheduler": ["```python"] + repr(lr_scheduler).splitlines() + ["```"],
        "Config": ["```yaml"]
        + config.dumps_yaml(sort_keys=False, indent=2).splitlines()
        + ["```"],
    }
    logger.log_summary(infos, sections, options)


def log_config(
    logger: lavd.Logger,
    config: ConfigEntry,
):
    if logger.disabled:
        return
    path = logger.get_file_path("config", extension=".yaml")
    config.save_yaml(path, sort_keys=False, indent=2)


def log_results(
    logger: lavd.Logger,
    epoch: int,
    train_result: Dict,
    validation_results: List[Dict],
    metrics: List[Metric],
):
    logger.log_scalar(train_result["lr"], "learning_rate", step=epoch)
    val = get_recursive(train_result, key="loss")
    logger.log_scalar(val, "train/loss", step=epoch)

    for result in validation_results:
        name = result["name"]
        for metric in metrics:
            key = metric.key
            val = get_recursive(result, key=key)
            logger.log_scalar(val, f"{name}/{key}", step=epoch)
        sample = get_recursive(result, key="sample")
        if sample:
            logger.log_image(sample["image"], f"{name}/sample", step=epoch)
            logger.log_text(
                sample["pred"], f"{name}/sample", step=epoch, expected=sample["text"]
            )


def log_epoch_stats(
    logger: lavd.Logger,
    results: List[Dict],
    metrics: List[Metric],
    lr_scheduler: Optional[BaseLrScheduler] = None,
    time_elapsed: Optional[float] = None,
):
    description = "{prefix}:".format(prefix=logger.prefix)
    if lr_scheduler is not None:
        description += " Learning Rate = {lr:.8f} ≈ {lr:.4e}".format(lr=lr_scheduler.lr)
        if lr_scheduler.is_warmup():
            description += " [🌡️ {step}/{warmup_steps} - {percent:.0%}]".format(
                step=lr_scheduler.step,
                warmup_steps=lr_scheduler.warmup_steps,
                percent=lr_scheduler.step / lr_scheduler.warmup_steps,
            )
    if time_elapsed is not None:
        description += " (time elapsed {elapsed})".format(
            elapsed=time.strftime("%H:%M:%S", time.gmtime(time_elapsed))
        )
    logger.println(description)
    header_names = ["Name"] + [metric.short_name or metric.name for metric in metrics]
    line_values = []
    for result in results:
        values = [result["name"]]
        for metric in metrics:
            val = get_recursive(result, key=metric.key)
            values.append(val)
        line_values.append(values)
    logger.print_table(header_names, line_values, indent_level=1)
