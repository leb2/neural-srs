import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from neural_srs.model.model import Model

from neural_srs.model.util import batch_data, load_data, BATCH_SIZE, NUM_FEATURES
import torch


def inference():
    sample_history = review_histories[1301]
    sample_history = np.array(sample_history)[np.newaxis]
    s = sample_history[0]

    colors = ['r', 'b', 'g', 'y']
    # colors = [(0, 0, 0)]
    estimates = []
    for i in range(sample_history.shape[1]):
        x = np.arange(-5, 20, 0.1)
        stability_estimate = model(torch.Tensor(sample_history[:, :i + 1]))[0, -1].item()
        y = np.exp(-np.exp(x) / (stability_estimate + 0.0000001))
        estimates.append(stability_estimate)

        print(stability_estimate)
        # plt.plot(x, y, c=colors[min(i, len(colors) - 1)])
        plt.plot(x, y, c=(1, .5, .5, (i + 1) / sample_history.shape[1]))
    estimates = np.array(estimates)
    v = np.concatenate((s, np.log(-np.log(0.9) * estimates[:, np.newaxis])), axis=-1)
    plt.ylim(0, 1)
    plt.show()

# for review in sample_history:
#     grade, duration, log_interval = review


#%% Correlation graph
def correlation_graph():
    batch_size2 = 1
    batches2, batched_labels2, batched_intervals2, masks2 \
        = batch_data(batch_size2, review_histories, intervals, labels)

    all_probabilities = []
    all_labels = []
    for index, (data, label, interval, mask) in enumerate(zip(batches2, batched_labels2, batched_intervals2, masks2)):
        data = torch.Tensor(data)
        label = torch.Tensor(label)
        interval = torch.Tensor(interval)
        stability_estimates = model(data)
        probabilities = torch.exp(-interval / torch.clamp(stability_estimates, min=0.000001))

        pr = np.squeeze(probabilities.detach().numpy(), axis=0)
        la = np.squeeze(label.detach().numpy(), axis=0)
        all_probabilities.append(pr)
        all_labels.append(la)

    all_probabilities = np.concatenate(all_probabilities)
    all_labels = np.concatenate(all_labels)

    sorted_probs, sorted_labels = zip(*sorted(zip(all_probabilities, all_labels)))
    sorted_probs = np.array(sorted_probs)
    sorted_labels = np.array(sorted_labels)


#%%
def plot_model():
    smoothed_labels = []
    smoothed_probs = []
    window = 500
    x_window = 500

    # bucket_size = 0.01
    # quartiles = np.arange(0, 1, bucket_size)
    # for quartile in quartiles:
    #     mask = np.logical_and(quartile < sorted_probs, sorted_probs <= (quartile + bucket_size))
    #     smoothed_probs.append(quartile + bucket_size / 2)
    #     smoothed_labels.append(np.mean(sorted_labels[mask]))

    for i in range(0, len(sorted_probs), 2 * x_window):
        smoothed_probs.append(np.mean(sorted_probs[max(i - x_window, 0): i+x_window]))
        smoothed_labels.append(np.mean(sorted_labels[max(i-window, 0):i+window]))

    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.scatter(smoothed_probs, smoothed_labels)
    plt.plot([0, 1], [0, 1], 'r')
    plt.show()


def train() -> None:
    reviews = load_data()
    batches = batch_data(reviews, BATCH_SIZE)
    num_batches = len(batches)
    num_train_batches = int(num_batches * 0.95)
    model = Model(input_dim=NUM_FEATURES, batch_size=BATCH_SIZE)

    opt = torch.optim.Adam(model.parameters(), lr=0.0001)
    for epoch in range(1, 100):
        total_train_loss = 0
        total_validation_loss = 0
        count_train = 0
        count_validation = 0

        for index, batch in tqdm(enumerate(batches), total=num_batches):
            x_data = batch.x_data
            y_data = batch.y_data
            intervals = batch.intervals
            mask = batch.mask

            x_data = torch.Tensor(x_data)
            y_data = torch.Tensor(y_data)
            intervals = torch.Tensor(intervals)

            da1 = x_data[0].detach().numpy()
            da2 = x_data[1].detach().numpy()

            la = y_data.detach().numpy()
            inter = intervals[0].detach().numpy()

            stability_estimates = model(x_data)
            probabilities = torch.exp(-intervals / torch.clamp(stability_estimates, min=0.000001))

            pr = probabilities.detach().numpy()
            s = stability_estimates.detach().numpy()

            criterion = torch.nn.BCELoss(weight=torch.Tensor(mask))
            loss = criterion(probabilities, y_data)

            criterion2 = torch.nn.BCELoss(weight=torch.Tensor(mask), reduce=False)
            ll = criterion2(probabilities, y_data).detach().numpy()

            if True or index < num_train_batches:
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

        print("Epoch %d, loss: %.3f, val loss: %.3f" % (epoch, average_train_loss, average_validation_loss))
        if epoch % 10 == 0:
            print("Saving model")
            torch.save(model.state_dict(), './weights.pth')
