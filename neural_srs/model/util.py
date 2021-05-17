from typing import NamedTuple, List


import numpy as np
from neural_srs.model.parse import Review


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
