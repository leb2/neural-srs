import sqlite3
from typing import List
import numpy as np
from dataclasses import dataclass


@dataclass
class Review:
    passed: bool
    log_interval: float


@dataclass
class ReviewData:
    review_histories:  List[List[Review]]
    labels: List


def load_data() -> ReviewData:
    conn = sqlite3.connect('../collection.anki2')
    c = conn.cursor()
    c.execute('''
        SELECT id, cid, ease, time FROM revlog ORDER BY cid, id;
    ''')

    review_histories = []
    labels = []
    intervals = []

    num_reviews = 0
    prev_review = None
    prev_interval = None

    curr_review_history = []
    curr_labels = []
    curr_intervals = []

    for review in c.fetchall():
        review = dict(zip(['time', 'card_id', 'grade', 'duration'], review))
        interval = 0 if prev_review is None else (review['time'] - prev_review['time']) / 60000
        if prev_review is None or review['card_id'] != prev_review['card_id']:
            if len(curr_review_history) != 0:
                review_histories.append(curr_review_history)
                intervals.append(curr_intervals)
                labels.append(curr_labels)
            interval = 0
            curr_review_history = []
            curr_labels = []
            curr_intervals = []
        else:
            curr_labels.append(int(review['grade'] != 1))
            curr_intervals.append(interval)
            curr_review_history.append([int(prev_review['grade'] > 1), np.log(prev_interval + 0.0001)])
        prev_review = review
        prev_interval = interval
        num_reviews += 1
    return review_histories, labels, intervals


def pad_batch(batch_x: List[float], batch_y: List[float], batch_intervals: List[float]):
    batch_size = len(batch_x)
    input_dim = len(batch_x[0][0])
    seq_len = np.max([len(review) for review in batch_x])

    padded = np.zeros((batch_size, seq_len, input_dim))
    padded_labels = np.zeros((batch_size, seq_len))
    padded_intervals = np.zeros((batch_size, seq_len))

    mask = np.zeros((batch_size, seq_len))
    for batch_index, (history, history_labels, history_intervals) in enumerate(zip(batch_x, batch_y, batch_intervals)):
        padded[batch_index, :len(history), :] = history
        padded_labels[batch_index, :len(history)] = history_labels
        padded_intervals[batch_index, :len(history)] = history_intervals
        mask[batch_index, :len(history)] = 1

    return padded, padded_labels, padded_intervals, mask


def batch_data(batch_size, review_histories, intervals, labels):
    batches = []
    batched_labels = []
    batched_intervals = []
    masks = []

    num_batches = len(review_histories) // batch_size

    for i in range(num_batches):
        start_index, end_index = i * batch_size, (i + 1) * batch_size
        batch_x = review_histories[start_index: end_index]
        batch_y = labels[start_index:end_index]
        batch_intervals = intervals[start_index:end_index]

        padded, padded_labels, padded_intervals, mask = pad_batch(batch_x, batch_y, batch_intervals)
        batches.append(padded)
        batched_labels.append(padded_labels)
        batched_intervals.append(padded_intervals)
        masks.append(mask)
    return batches, batched_labels, batched_intervals, masks
