import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

n = 1000
true_d = 50

X = torch.randn(n, true_d)
theta_star = torch.randn(true_d, 1)
y = X @ theta_star   # noiseless linear model


def expand_features(X, d):
    n, d0 = X.shape
    if d == d0:
        return X
    elif d < d0:
        return X[:, :d]
    else:
        pad = torch.randn(n, d - d0) * 0.1
        return torch.cat([X, pad], dim=1)

class LinearModel(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(d, 1))

    def forward(self, x):
        return x @ self.w

def train_sgd(X, y, d, lr=0.01, epochs=10000):
    Xd = expand_features(X, d)
    model = LinearModel(d)
    opt = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    norms = []

    for epoch in range(epochs):
        # SGD step (batch size = 1)
        idx = torch.randint(0, Xd.shape[0], (1,))
        xb = Xd[idx]
        yb = y[idx]

        loss = loss_fn(model(xb), yb)

        opt.zero_grad()
        loss.backward()
        opt.step()

        norms.append(model.w.norm().item())

    return norms

d_list = [10, 50, 100, 2000] 

plt.figure(figsize=(10,6))

for d in d_list:
    norms = train_sgd(X, y, d)
    plt.plot(norms, label=f"d={d}")

plt.xlabel("Epoch")
plt.ylabel("||w||2")
plt.title("SGD implicit bias: norm evolution vs epochs")
plt.legend()
plt.grid()
plt.savefig("norm_convergence.png")