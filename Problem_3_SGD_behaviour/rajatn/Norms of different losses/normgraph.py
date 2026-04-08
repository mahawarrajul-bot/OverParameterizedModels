import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification


torch.manual_seed(0)
np.random.seed(0)


X, y = make_classification(
    n_samples=2000,
    n_features=20,
    n_informative=20,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=3.0,
    flip_y=0,
)

y = 2 * y - 1

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

dataset = torch.utils.data.TensorDataset(X, y)
loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)


class LinearModel(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(d))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x @ self.w + self.b



# Loss functions

def logistic_loss(y, f):
    return torch.log(1 + torch.exp(-y * f)).mean()

def exponential_loss(y, f):
    return torch.exp(-y * f).mean()

def hinge_loss(y, f):
    return torch.clamp(1 - y * f, min=0).mean()

def squared_loss(y, f):
    return ((y - f) ** 2).mean()



def train(loss_fn):

    model = LinearModel(20)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    norms = []

    for epoch in range(1000):

        for xb, yb in loader:

            f = model(xb)
            loss = loss_fn(yb, f)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            norm = torch.norm(model.w).item()
            norms.append(norm)

    return norms



logistic_norm = train(logistic_loss)
exp_norm = train(exponential_loss)
hinge_norm = train(hinge_loss)
sq_norm = train(squared_loss)



plt.figure(figsize=(8,6))

plt.plot(logistic_norm, label="Logistic Loss")


plt.xlabel("Epoch")
plt.ylabel("||w||")
plt.title("Weight Norm Growth for Different Loss Functions")
plt.legend()
plt.grid(True)

plt.savefig("Weight Norm Growth for Different Loss Functions Logistic Loss")




plt.figure(figsize=(8,6))

plt.plot(exp_norm, label="Exponential Loss")


plt.xlabel("Epoch")
plt.ylabel("||w||")
plt.title("Weight Norm Growth for Different Loss Functions")
plt.legend()
plt.grid(True)

plt.savefig("Weight Norm Growth for Different Loss Functions Exponential Loss")

plt.figure(figsize=(8,6))

plt.plot(hinge_norm, label="Hinge Loss")


plt.xlabel("Epoch")
plt.ylabel("||w||")
plt.title("Weight Norm Growth for Different Loss Functions")
plt.legend()
plt.grid(True)

plt.savefig("Weight Norm Growth for Different Loss Functions Hinge Loss")

plt.figure(figsize=(8,6))


plt.plot(sq_norm, label="Squared Loss")


plt.xlabel("Epoch")
plt.ylabel("||w||")
plt.title("Weight Norm Growth for Different Loss Functions")
plt.legend()
plt.grid(True)

plt.savefig("Weight Norm Growth for Different Loss Functions Squared Loss")