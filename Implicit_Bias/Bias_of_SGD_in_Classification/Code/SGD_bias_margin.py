import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.svm import SVC

torch.manual_seed(0)

def generate_data(n, d_true):
    X = torch.randn(n, d_true)
    true_w = torch.randn(d_true)
    y = torch.where(X @ true_w >= 0, 1.0, -1.0)
    return X, y

def pad_data(X, d_new):
    n, d_old = X.shape
    if d_new > d_old:
        pad = torch.randn(n, d_new - d_old)*0.01
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
        if epoch % 10 == 0:
            margin = compute_margin(w.detach(), Xd, y)
            margins.append(margin)
    return margins

def get_hard_margin(X, y, d):
    Xd = pad_data(X, d)
    clf = SVC(kernel='linear', C=1e6)
    clf.fit(Xd.numpy(), y.numpy())
    w_star = torch.tensor(clf.coef_[0], dtype=torch.float32)
    return compute_margin(w_star, Xd, y)

n = 100
d_true = 20

X, y = generate_data(n, d_true)

d_values = [10, 100, 500]

plt.figure(figsize=(8, 5))
colors = {
    10: "lightgreen",
    100: "red",
    500: "skyblue"
}

for d in d_values:
    margins = train_and_track_margin(X, y, d)
    svm_margin = get_hard_margin(X, y, d)

    print(f"d={d}, Final GD margin: {margins[-1]:.4f}")
    print(f"d={d}, SVM margin: {svm_margin:.4f}")

    color = colors[d]

    plt.plot(margins, label=f"GD d={d}", color=color)
    plt.axhline(y=svm_margin, linestyle='--', color=color, label=f"SVM d={d}")

plt.xlabel("Epoch / 10")
plt.ylabel("Margin")
plt.title("Margin vs Epochs (GD → Max-Margin)")
plt.legend()
plt.grid()
plt.savefig("margin_plot_final.png")