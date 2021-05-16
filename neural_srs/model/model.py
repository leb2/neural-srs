import torch


class Model(torch.nn.Module):
    def __init__(self, input_dim, batch_size, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, 1)
        self.batch_size = batch_size

        hidden_shape = (1, 1, self.hidden_dim)
        self.hidden1 = torch.nn.Parameter(torch.randn(*hidden_shape) / 100)
        self.hidden2 = torch.nn.Parameter(torch.randn(*hidden_shape) / 100)

    def forward(self, batch):
        batch_size = batch.shape[0]
        initial_hidden = (self.hidden1.repeat(1, batch_size, 1), self.hidden2.repeat(1, batch_size, 1))

        # h1 = initial_hidden[0][0].detach().numpy()
        # h2 = initial_hidden[1][0].detach().numpy()
        # b = batch[0].detach().numpy()
        out, hidden = self.lstm(batch, initial_hidden)

        stability_estimate = self.linear1(out)
        stability_estimate = torch.squeeze(self.head(stability_estimate), dim=-1)

        # c = stability_estimate.detach().numpy()
        return torch.exp(stability_estimate)
