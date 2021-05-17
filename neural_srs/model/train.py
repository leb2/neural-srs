from typing import List

from tqdm import tqdm

from neural_srs.model.constants import DEFAULT_SAVE_PATH, BATCH_SIZE
from neural_srs.model.model import Model

import torch

from neural_srs.model.parse import batch_data
from neural_srs.model.types import Review


def train_model(model: Model, reviews: List[List[Review]], save_path: str = DEFAULT_SAVE_PATH) -> Model:
    batches = batch_data(reviews, BATCH_SIZE)

    num_batches = len(batches)
    num_train_batches = int(num_batches * 0.95)
    model.batches = batches

    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(1, 10):
        total_train_loss = 0
        total_validation_loss = 0
        count_train = 0
        count_validation = 0
        all_predictions = []

        for index, batch in tqdm(enumerate(batches), total=num_batches):
            x_data = torch.Tensor(batch.x_data)
            y_data = torch.Tensor(batch.y_data)
            intervals = torch.Tensor(batch.intervals)
            probabilities = model(x_data, intervals)
            all_predictions.append(probabilities)

            criterion = torch.nn.BCELoss(weight=torch.Tensor(batch.mask))
            loss = criterion(probabilities, y_data)

            if index < num_train_batches:
                opt.zero_grad()
                loss.backward()
                opt.step()

                total_train_loss += loss
                count_train += 1
            else:
                total_validation_loss += loss
                count_validation += 1

        average_train_loss = total_train_loss / count_train
        average_validation_loss = 0  # total_validation_loss / count_validation

        print("Epoch %d, loss: %.3f, validation loss: %.3f" % (epoch, average_train_loss, average_validation_loss))
        model.predictions = all_predictions
        model.save(save_path)
    return model
