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

    num_reviews = 0
    prev_review = None
    curr_review_history = []
    curr_labels = []

    for review in c.fetchall():
        review = dict(zip(['time', 'card_id', 'grade', 'duration'], review))

        if prev_review is None or review['card_id'] != prev_review['card_id']:
            if len(curr_review_history) != 0:
                review_histories.append(curr_review_history)
                labels.append(curr_labels)
            curr_review_history = []
            curr_labels = []
        else:
            # TODO: one hot grade
            interval = np.log((review['time'] - prev_review['time']) / 60000)
            curr_review_history.append([int(prev_review['grade'] > 2), prev_review['duration'], interval])
            curr_labels.append(int(review['grade'] != 1))
        prev_review = review
        num_reviews += 1
    return review_histories, labels


def pad_batch(batch_x, batch_y):
    batch_size = len(batch_x)
    input_dim = len(batch_x[0][0])
    seq_len = np.max([len(review) for review in batch_x])

    padded = np.zeros((batch_size, seq_len, input_dim))
    padded_labels = np.zeros((batch_size, seq_len))
    mask = np.zeros((batch_size, seq_len))
    for batch_index, (history, history_labels) in enumerate(zip(batch_x, batch_y)):
        padded[batch_index, :len(history), :] = history
        padded_labels[batch_index, :len(history)] = history_labels
        mask[batch_index, :len(history)] = 1

    return padded, padded_labels, mask


def batch_data(batch_size, review_histories, labels):
    batches = []
    batched_labels = []
    masks = []

    num_batches = len(review_histories) // batch_size

    for i in range(num_batches):
        start_index, end_index = i * batch_size, (i + 1) * batch_size
        batch_x = review_histories[start_index: end_index]
        batch_y = labels[start_index:end_index]

        padded, padded_labels, mask = pad_batch(batch_x, batch_y)
        batches.append(padded)
        batched_labels.append(padded_labels)
        masks.append(mask)
    return batches, batched_labels, masks


batch_size = 4
review_histories, labels = load_data()
input_dim = len(review_histories[0][0])
batches, batched_labels, masks = batch_data(batch_size, review_histories, labels)


#%% Create model

class Model(torch.nn.Module):
    def __init__(self, input_dim, batch_size, hidden_dim=64):
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
        out, hidden = self.lstm(batch, initial_hidden)
        logits = torch.squeeze(self.head(out), dim=-1)
        return logits


model = Model(input_dim, batch_size)
# model.load_state_dict(torch.load('./weights.pth'))


#%% Training

opt = torch.optim.Adam(model.parameters(), lr=0.0001)
for epoch in range(1, 5):
    total_loss = 0
    count = 0
    for index, (batch, label, mask) in enumerate(zip(batches, batched_labels, masks)):
        batch = torch.Tensor(batch)
        label = torch.Tensor(label)

        logits = model(batch)
        criterion = torch.nn.BCEWithLogitsLoss(weight=torch.Tensor(mask))
        loss = criterion(logits, label)

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
sample_history = np.array(sample_history)[:6, np.newaxis]

log_time = np.arange(-5, 12, 0.05)
y = []
for time in log_time:
    time_modified = sample_history
    time_modified[-1, 0, -1] = time
    guess = F.sigmoid(model(torch.Tensor(time_modified))[0, -1]).item()
    y.append(guess)

print(y)
plt.ylim(0, 1)
plt.plot(log_time, y)
plt.show()

# for review in sample_history:
#     grade, duration, log_interval = review























