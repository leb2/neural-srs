from typing import Optional, List, Tuple
import numpy as np

from neural_srs.model.constants import SAVE_ROOT_DIR, DEFAULT_SAVE_PATH
from neural_srs.model.types import DataBatch, FlattenedY, CardReviewHistory
import torch
from os import path
from tqdm import tqdm


class Model(torch.nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim=64,
            save_path=str,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, 2)

        hidden_shape = (1, 1, self.hidden_dim)
        self.hidden1 = torch.nn.Parameter(torch.randn(*hidden_shape) / 100)
        self.hidden2 = torch.nn.Parameter(torch.randn(*hidden_shape) / 100)

        self.batches: Optional[List[DataBatch]] = None
        self.predictions: Optional[List[torch.Tensor]] = None  # list of tensors of shape (batch_size, sequence_length)
        self.save_path = save_path
        self._flattened_y: Optional[FlattenedY] = None
        self.reviews: Optional[List[CardReviewHistory]] = None

    def predict_head(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param batch: Tensor of shape (num_sequences, sequence_length, num_features)
        :return:
            stability_estimate: tensor of shape (num_sequences, sequence_length)
            base_fail_rate: tensor of shape (num_sequences, sequence_length)
        """
        batch_size = batch.shape[0]
        initial_hidden = (self.hidden1.repeat(1, batch_size, 1), self.hidden2.repeat(1, batch_size, 1))
        out, hidden = self.lstm(batch, initial_hidden)

        prediction = self.linear1(out)
        prediction = self.head(prediction)

        stability_estimate = prediction[:, :, 0]
        base_fail_rate = torch.sigmoid(prediction[:, :, 1])

        stability_estimate = torch.clamp(torch.exp(stability_estimate), min=0.000001)
        return stability_estimate, base_fail_rate

    def stability_estimates(self, batch: torch.Tensor):
        """
        :param batch: Tensor of shape (num_sequences, sequence_length, num_features)
        :return: Tensor of shape (num_sequences, sequence_length)
        """
        stability_estimates, _ = self.predict_head(batch)
        return stability_estimates

    def forward(self, batch: torch.Tensor, intervals: torch.Tensor) -> torch.Tensor:
        """
        :param batch: Tensor of shape (num_sequences, sequence_length, num_features)
        :param intervals: Tensor of shape (num_sequences, sequence_length)
        :return: Tensor of shape (num_sequences, sequence_length)
        """
        stability_estimates, base_fail_rate = self.predict_head(batch)
        probabilities = (1 - base_fail_rate) * torch.exp(-intervals / stability_estimates)
        return probabilities

    @property
    def flattened_y(self) -> FlattenedY:
        y = []
        y_hat = []
        for batch, predictions in zip(self.batches, self.predictions):
            batch_size = len(batch.x_data)
            for i in range(batch_size):
                sequence_length = int(batch.mask[i].sum().item())
                y += batch.y_data[i, :sequence_length].tolist()
                y_hat += predictions[i, :sequence_length].tolist()
        self._flattened_y = FlattenedY(y=np.array(y), y_hat=np.array(y_hat))
        return self._flattened_y

    def compute_estimates(self) -> None:
        """
        Computes stability and base_fail_rate estimates for each review, and saves it self.reviews
        """
        print("Computing estimates")
        for batch_num, batch in tqdm(enumerate(self.batches), total=len(self.batches)):
            batch_size = len(batch.x_data)
            stability_estimates, base_fail_rates = self.predict_head(batch.x_data)

            for i in range(batch_size):
                card_index = batch_num * batch_size + i
                sequence_length = int(batch.mask[i].sum().item())
                for sequence_index in range(sequence_length):
                    review = self.reviews[card_index].reviews[sequence_index]
                    review.stability_estimate = stability_estimates[i, sequence_index].item()
                    review.base_fail_rate_estimate = base_fail_rates[i, sequence_index].item()
        self.save()

    def save(self, save_name: str = DEFAULT_SAVE_PATH) -> None:
        torch.save(self, path.join(SAVE_ROOT_DIR, save_name))

    @classmethod
    def load(cls, save_name: str = DEFAULT_SAVE_PATH) -> 'Model':
        model = torch.load(path.join(SAVE_ROOT_DIR, save_name))
        model.eval()
        return model
