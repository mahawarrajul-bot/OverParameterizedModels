import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(0)

def generate_data(n, d_true):
    X = torch.randn(n, d_true)
    true_w = torch.randn(d_true)
    y = torch.where(X @ true_w >= 0, 1.0, -1.0)
    return X, y


def pad_data(X, d_new):
    n, d_old = X.shape
    if d_new > d_old:
        pad = torch.randn(n, d_new - d_old) * 0.1  # small noise
        return torch.cat([X, pad], dim=1)
    else:
        return X[:, :d_new]

def compute_margin(w, X, y):
    logits = X @ w
    margins = y * logits
    return margins.min().item() / (w.norm().item() + 1e-12)

def train_and_track_margin(X, y, d, epochs=10000, lr=0.1):
    Xd = pad_data(X, d)

    w = torch.zeros(d, requires_grad=True)
    optimizer = optim.SGD([w], lr=lr)

    margins = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        logits = Xd @ w
        loss = torch.mean(F.softplus(-y * logits)) 

        loss.backward()
        optimizer.step()

        margin = compute_margin(w.detach(), Xd, y)
        margins.append(margin)

    return margins

n = 100
d_true = 20

X, y = generate_data(n, d_true)

d_values = [10, 100, 500, 1000]

plt.figure(figsize=(8, 5))

for d in d_values:
    margins = train_and_track_margin(X, y, d)
    print(f"Final margin for d={d}: {margins[-1]:.4f}")
    plt.plot(margins, label=f"d={d}")

plt.xlabel("Epoch")
plt.ylabel("Margin")
plt.title("Margin vs Epochs")
plt.legend()
plt.grid()

plt.savefig("margin_plot_diff_d.png")
