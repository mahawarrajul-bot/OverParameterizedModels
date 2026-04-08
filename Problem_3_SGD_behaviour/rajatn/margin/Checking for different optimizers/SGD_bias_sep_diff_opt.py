import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification

torch.manual_seed(0)
np.random.seed(0)

#generate perfectly separable data
X, y = make_classification(
    n_samples=2000,
    n_features=20,
    n_informative=20,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=3.0,
    flip_y=0,
)

#convert labels to {-1,1}
y = 2 * y - 1

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

dataset = torch.utils.data.TensorDataset(X, y)
loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)


class DeepLinear(nn.Module):

    def __init__(self, d, depth=5):
        super().__init__()

        self.layers = nn.ModuleList(
            [nn.Linear(d, d, bias=False) for _ in range(depth)]
        )

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x.sum(dim=1)


def effective_weight(model):

    W = model.layers[0].weight.data

    for layer in model.layers[1:]:
        W = layer.weight.data @ W

    ones = torch.ones(W.shape[0])
    w = W.t() @ ones

    return w


def compute_margin(model, X, y):

    w = effective_weight(model)

    logits = X @ w

    margins = y * logits

    margin = torch.min(margins) / torch.norm(w)

    return margin.item()


def weight_norm(model):

    w = effective_weight(model)

    return torch.norm(w).item()


def train(optimizer_name):

    model = DeepLinear(X.shape[1], depth=5)

    if optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=0.01)

    elif optimizer_name == "momentum":
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    elif optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=0.01)

    loss_fn = nn.BCEWithLogitsLoss()

    margins = []
    norms = []

    for epoch in range(1000):

        for xb, yb in loader:

            optimizer.zero_grad()

            logits = model(xb)

            loss = loss_fn(logits, (yb + 1) / 2)

            loss.backward()

            optimizer.step()

        margins.append(compute_margin(model, X, y))
        norms.append(weight_norm(model))

    return margins, norms


sgd_margin, sgd_norm = train("sgd")
momentum_margin, momentum_norm = train("momentum")
adam_margin, adam_norm = train("adam")



plt.figure()

plt.plot(sgd_margin, label="SGD")
plt.plot(momentum_margin, label="SGD + Momentum")
plt.plot(adam_margin, label="Adam")

plt.xlabel("Epoch")
plt.ylabel("Margin")

plt.legend()
plt.title("Margin Growth")

plt.savefig("margin_growth.png")

