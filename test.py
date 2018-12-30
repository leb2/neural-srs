import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn((1,)))

    def forward(self):
        return self.a


model = Model()
opt = torch.optim.Adam(model.parameters(), lr=0.0001)

for i in range(2):
    loss = model()
    opt.zero_grad()
    loss.backward()
    opt.step()

print("done")


