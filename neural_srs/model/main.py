from neural_srs.model.constants import NUM_FEATURES
from neural_srs.model.model import Model
from neural_srs.model.metrics import print_metrics, format_estimates, calibration_plot
import torch

from neural_srs.model.train import train_model


def main():
    print("cuda is available", torch.cuda.is_available())

    model = Model(input_dim=NUM_FEATURES)
    model = Model.load()

    # train_model(model, num_epochs=15)
    # model.compute_estimates()

    format_estimates(model)
    # print_metrics(model)
    # calibration_plot(model)


if __name__ == '__main__':
    main()