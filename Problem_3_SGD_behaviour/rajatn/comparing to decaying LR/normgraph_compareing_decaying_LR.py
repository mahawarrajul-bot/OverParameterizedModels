import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification



torch.manual_seed(0)
np.random.seed(0)



X, y = make_classification(
    n_samples=20000,
    n_features=20,
    n_informative=20,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=3.0,
    flip_y=0,
)

y = 2*y - 1

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

dataset = torch.utils.data.TensorDataset(X, y)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    shuffle=True
)



class TwoLayerLinear(nn.Module):

    def __init__(self, input_dim, hidden_dim, seed=0):

        super().__init__()

        g = torch.Generator()
        g.manual_seed(seed)

        self.W1 = nn.Parameter(torch.randn(input_dim, hidden_dim, generator=g) * 0.01)
        self.b1 = nn.Parameter(torch.randn(hidden_dim, generator=g) * 0.01)

        self.W2 = nn.Parameter(torch.randn(hidden_dim, 1, generator=g) * 0.01)
        self.b2 = nn.Parameter(torch.randn(1, generator=g) * 0.01)

    def forward(self, x):

        h = x @ self.W1 + self.b1
        f = h @ self.W2 + self.b2

        return f.squeeze()



def logistic_loss(y, f):
    return torch.log(1 + torch.exp(-y*f)).mean()


def exponential_loss(y, f):
    return torch.exp(-y*f).mean()


def hinge_loss(y, f):
    return torch.clamp(1 - y*f, min=0).mean()


def squared_loss(y, f):
    return ((y - f)**2).mean()



def train(loss_fn, hidden_dim, lambda_reg=1e-4, decay_lr=False):

    model = TwoLayerLinear(20, hidden_dim, seed=42)

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    if decay_lr:
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 1/(1 + 0.05*epoch)
        )

    norms = []

    for epoch in range(300):

        for xb, yb in loader:

            f = model(xb)

            loss = loss_fn(yb, f)

            l2_reg = (model.W1**2).sum() + (model.W2**2).sum()
            loss += lambda_reg * l2_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if decay_lr:
            scheduler.step()

        with torch.no_grad():
            norm = torch.sqrt((model.W1**2).sum() + (model.W2**2).sum())
            norms.append(norm.item())

    return norms


hidden_sizes = [50, 500, 1000, 2000]


losses = {
    "Logistic": logistic_loss,
    "Exponential": exponential_loss,
    "Hinge": hinge_loss,
    "Squared": squared_loss,
}



hidden_sizes = [50, 500, 1000, 2000]

for loss_name, loss_fn in losses.items():

    plt.figure(figsize=(8,6))

    # constant LR for multiple widths
    for h in hidden_sizes:
        norms = train(loss_fn, h, decay_lr=False)
        plt.plot(norms, label=f"Const LR, hidden={h}")

    # decaying LR for only width = 5000
    norms_decay = train(loss_fn, 5000, decay_lr=True)
    plt.plot(norms_decay, "--", linewidth=3, label="Decay LR, hidden=5000")

    plt.xlabel("Epoch")
    plt.ylabel("||W||")
    plt.title(f"{loss_name} Loss : Norm Growth")
    plt.legend()
    plt.grid(True)

    plt.savefig(f"{loss_name}_norm_comparison.png")