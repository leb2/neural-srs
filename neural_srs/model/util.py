from typing import NamedTuple, List
import datetime
import re

import numpy as np

from neural_srs.model.types import Review


class VectorizedReviews(NamedTuple):
    x_data: np.ndarray
    y_data: np.ndarray
    intervals: np.ndarray


def vectorize_reviews(reviews: List[Review]) -> VectorizedReviews:
    x_data = []
    y_data = []
    intervals = []
    for review in reviews:
        x_data.append([review.passed, review.prev_log_interval])
        y_data.append(review.label)
        intervals.append(review.next_interval)
    return VectorizedReviews(np.array(x_data), np.array(y_data), np.array(intervals))


def strip_tags(text: str) -> str:
    return re.sub('<[^<]+?>', '', text)


def stability_to_interval(stability: float, target: float = 0.9) -> str:
    interval = -np.log(target) * stability
    time_delta = datetime.timedelta(minutes=interval)
    return str(time_delta)
