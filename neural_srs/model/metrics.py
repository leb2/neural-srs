from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from neural_srs.model.model import Model

import torch

from neural_srs.model.parse import batch_data
from neural_srs.model.types import Review
from neural_srs.model.util import vectorize_reviews
from sklearn.calibration import calibration_curve
import sklearn


def print_metrics(model: Model) -> None:
    y = model.flattened_y.y
    y_hat = model.flattened_y.y_hat

    accuracy = np.sum((y_hat > 0.6) == y) / len(y_hat)
    print("accuracy", accuracy)

    roc = roc_auc_score(y, y_hat, average=None)
    print("roc score", roc)

    plot_roc_curve(model)


def inference(model: Model, reviews: List[Review]) -> None:
    vectorized_reviews = vectorize_reviews(reviews)
    sample_history = vectorized_reviews.x_data[np.newaxis]

    estimates = []
    for i in range(sample_history.shape[1]):
        x = np.arange(0, 100000, 60)

        stability_estimate = model.stability_estimates(torch.Tensor(sample_history[:, :i + 1]))[0, -1].item()
        y = np.exp(-x / stability_estimate)

        estimates.append(stability_estimate)
        plt.plot(x, y, c=(1, .5, .5, (i + 1) / sample_history.shape[1]))
    plt.ylim(0, 1)
    plt.show()


def plot_roc_curve(model: Model) -> None:
    y = model.flattened_y.y
    y_hat = model.flattened_y.y_hat

    false_positive_rate, true_positive_rate, thresholds = sklearn.metrics.roc_curve(y + 1, y_hat, pos_label=2)
    plt.plot(false_positive_rate, true_positive_rate, 'b')
    plt.plot([0, 1], [0, 1], 'r')
    plt.show()


def calibration_plot(model: Model) -> None:
    flattened_y = model.flattened_y
    y = flattened_y.y
    y_hat = flattened_y.y_hat

    sorted_probs, sorted_labels = zip(*sorted(zip(y_hat, y)))
    sorted_probs = np.array(sorted_probs)
    sorted_labels = np.array(sorted_labels)

    smoothed_labels = []
    smoothed_probs = []
    window = 500
    x_window = 500

    for i in range(0, len(sorted_probs), 2 * x_window):
        smoothed_probs.append(np.mean(sorted_probs[max(i - x_window, 0): i+x_window]))
        smoothed_labels.append(np.mean(sorted_labels[max(i-window, 0):i+window]))

    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.scatter(smoothed_probs, smoothed_labels)
    plt.plot([0, 1], [0, 1], 'r')

    prob_true, prob_pred = calibration_curve(y, y_hat, n_bins=100)
    plt.plot(prob_pred, prob_true, 'b')
    plt.show()

