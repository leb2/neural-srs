import sqlite3
from typing import List, Tuple
import numpy as np
from dataclasses import dataclass


BATCH_SIZE = 20


NUM_FEATURES = 2


@dataclass
class Review:
    passed: int
    label: int
    prev_log_interval: float
    next_log_interval: float


def load_data() -> List[List[Review]]:
    conn = sqlite3.connect('../collection.anki2')
    c = conn.cursor()
    c.execute('''
        SELECT id, cid, ease, time FROM revlog ORDER BY cid, id;
    ''')

    prev_review = None
    prev_interval = None

    reviews = []
    curr_reviews = []

    for review in c.fetchall():
        review = dict(zip(['time', 'card_id', 'grade', 'duration'], review))
        interval = 0 if prev_review is None else (review['time'] - prev_review['time']) / 60000
        if prev_review is None or review['card_id'] != prev_review['card_id']:
            if len(curr_reviews) != 0:
                reviews.append(curr_reviews)
            interval = 0
            curr_reviews = []
        else:
            curr_reviews.append(Review(
                prev_log_interval=np.log(prev_interval + 0.00001),
                passed=int(prev_review['grade'] > 1),
                next_log_interval=interval,
                label=int(review['grade'] != 1),
            ))
        prev_review = review
        prev_interval = interval
    return reviews


def _vectorize_reviews(reviews: List[Review]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_data = []
    y_data = []
    intervals = []
    for review in reviews:
        x_data.append([review.passed, review.prev_log_interval])
        y_data.append(review.label)
        intervals.append(review.next_log_interval)
    return np.array(x_data), np.array(y_data), np.array(intervals)


def _pad_batch(all_reviews: List[List[Review]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    batch_size = len(all_reviews)
    seq_len = np.max([len(review) for review in all_reviews])

    padded_x = np.zeros((batch_size, seq_len, NUM_FEATURES))
    padded_y = np.zeros((batch_size, seq_len))
    padded_intervals = np.zeros((batch_size, seq_len))

    mask = np.zeros((batch_size, seq_len))
    for batch_index, review_chain in enumerate(all_reviews):
        x_data, y_data, intervals = _vectorize_reviews(review_chain)

        padded_x[batch_index, :len(x_data), :] = x_data
        padded_y[batch_index, :len(x_data)] = y_data
        padded_intervals[batch_index, :len(x_data)] = intervals
        mask[batch_index, :len(x_data)] = 1

    return padded_x, padded_y, padded_intervals, mask


@dataclass
class DataBatch:
    x_data: np.ndarray
    y_data: np.ndarray
    intervals: np.ndarray
    mask: np.ndarray


def batch_data(
        reviews: List[List[Review]],
        batch_size: int,
) -> List:
    batches = []
    num_batches = len(reviews) // batch_size

    for i in range(num_batches):
        start_index, end_index = i * batch_size, (i + 1) * batch_size
        batch = reviews[start_index: end_index]

        padded_x, padded_y, padded_intervals, mask = _pad_batch(batch)
        batches.append(DataBatch(
            x_data=padded_x,
            y_data=padded_y,
            intervals=padded_intervals,
            mask=mask
        ))
    return batches
