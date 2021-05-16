from neural_srs.model.model import Model
from neural_srs.model.train import train
from neural_srs.model.util import load_data, batch_data


def main():
    model = train()
    print(model)


if __name__ == '__main__':
    main()