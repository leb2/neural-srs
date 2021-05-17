from dataclasses import dataclass
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


@dataclass
class FlattenedY:
    y: np.ndarray
    y_hat: np.ndarray
