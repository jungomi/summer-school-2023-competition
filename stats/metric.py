from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .dict_utils import get_recursive


@dataclass
class Metric:
    name: str
    key: str
    order: str
    short_name: Optional[str] = None


METRICS = [
    Metric(
        key="cer",
        name="Character Error Rate (CER)",
        short_name="CER",
        order="min",
    ),
    Metric(
        key="wer",
        name="Word Error Rate (WER)",
        short_name="WER",
        order="min",
    ),
]
# For convenience to index by key
METRICS_DICT = {m.key: m for m in METRICS}


# Calculates the average of the metric across all validation sets
def average_checkpoint_metric(results: List[Dict], key: str = "loss") -> float:
    values_per_result = []
    for result in results:
        val = get_recursive(result, key=key)
        values_per_result.append(val)
    # Take the mean of all results
    value = torch.mean(torch.tensor(values_per_result, dtype=torch.float), dim=0)
    return float(value)
