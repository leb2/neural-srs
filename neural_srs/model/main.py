from neural_srs.model.constants import NUM_FEATURES
from neural_srs.model.model import Model
from neural_srs.model.parse import load_data
from neural_srs.model.train import train_model
from neural_srs.model.metrics import inference, calibration_plot, print_metrics
import torch


def main():
    print("cuda is available", torch.cuda.is_available())

    # model = Model(input_dim=NUM_FEATURES)
    model = Model.load()
    train_model(model, reviews)
    print_metrics(model)

    # inference(model, reviews[0])
    # calibration_plot(model)


if __name__ == '__main__':
    main()