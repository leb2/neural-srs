import sqlite3
import numpy as np
import torch
import torch.nn.functional as F


#%% Pre-process

def load_data():
    conn = sqlite3.connect('./collection.anki2')
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


def pad_batch(batch_x, batch_y, batch_intervals):
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


batch_size = 2
review_histories, labels, intervals = load_data()
input_dim = len(review_histories[0][0])
batches, batched_labels, batched_intervals, masks = batch_data(batch_size, review_histories, intervals, labels)


#%% Create model

class Model(torch.nn.Module):
    def __init__(self, input_dim, batch_size, hidden_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = torch.nn.Linear(hidden_dim, 1)
        self.batch_size = batch_size

        hidden_shape = (1, 1, self.hidden_dim)
        self.hidden1 = torch.nn.Parameter(torch.randn(*hidden_shape) / 100)
        self.hidden2 = torch.nn.Parameter(torch.randn(*hidden_shape) / 100)

    def forward(self, batch):
        batch_size = batch.shape[0]
        initial_hidden = (self.hidden1.repeat(1, batch_size, 1), self.hidden2.repeat(1, batch_size, 1))

        h1 = initial_hidden[0][0].detach().numpy()
        h2 = initial_hidden[1][0].detach().numpy()
        b = batch[0].detach().numpy()
        b2 = batch[1].detach().numpy()
        out, hidden = self.lstm(batch, initial_hidden)
        stability_estimate = torch.squeeze(self.head(out), dim=-1)
        c = stability_estimate.detach().numpy()
        return F.relu(stability_estimate)


model = Model(input_dim, batch_size)
# model.load_state_dict(torch.load('./weights.pth'))


#%% Training

opt = torch.optim.RMSprop(model.parameters(), lr=0.0001)
for epoch in range(1, 100):
    total_loss = 0
    count = 0
    for index, (data, label, interval, mask) in enumerate(zip(batches, batched_labels, batched_intervals, masks)):
        data = torch.Tensor(data)
        label = torch.Tensor(label)
        interval = torch.Tensor(interval)

        da1 = data[0].detach().numpy()
        da2 = data[1].detach().numpy()

        la = label.detach().numpy()
        inter = interval[0].detach().numpy()

        stability_estimates = model(data)
        probabilities = torch.exp(-interval / torch.clamp(stability_estimates, min=0.000001))
        pr = probabilities.detach().numpy()

        s = stability_estimates.detach().numpy()
        # loss = torch.mean(torch.pow((probabilities - label) * torch.Tensor(mask), 2))
        criterion = torch.nn.BCELoss(weight=torch.Tensor(mask))
        loss = criterion(probabilities, label)

        criterion2 = torch.nn.BCELoss(weight=torch.Tensor(mask), reduce=False)
        ll = criterion2(probabilities, label).detach().numpy()

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss
        count += 1
    average_loss = total_loss / count
    print("Epoch %d, loss: %.3f" % (epoch, average_loss))
torch.save(model.state_dict(), './weights.pth')


#%% Inference
import matplotlib.pyplot as plt


sample_history = review_histories[1300]
sample_history = np.array(sample_history)[np.newaxis]

colors = ['r', 'b', 'g', 'y']
for i in range(sample_history.shape[1]):
    x = np.arange(0, 1000)
    stability_estimate = model(torch.Tensor(sample_history[:, :i + 1]))[0, -1].item()
    y = np.exp(-x / (stability_estimate + 0.0000001))

    print(stability_estimate)
    plt.plot(x, y, colors[min(i, len(colors) - 1)])
plt.ylim(0, 1)
plt.show()

# for review in sample_history:
#     grade, duration, log_interval = review























