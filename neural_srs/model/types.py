from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch


@dataclass
class DataBatch:
    x_data: torch.Tensor  # tensor of shape (batch_size, sequence_length, num_features)
    y_data: torch.Tensor  # tensor of shape (batch_size, sequence_length)
    intervals: torch.Tensor  # tensor of shape (batch_size, sequence_length)
    mask: torch.Tensor  # tensor of shape (batch_size, sequence_length)


@dataclass
class Review:
    passed: int
    label: int
    prev_log_interval: float
    next_interval: float

    stability_estimate: Optional[float] = None
    base_fail_rate_estimate: Optional[float] = None


@dataclass()
class CardReviewHistory:
    reviews: List[Review] = field(default_factory=lambda: [])
    word: Optional[str] = None


@dataclass
class FlattenedY:
    y: np.ndarray
    y_hat: np.ndarray
