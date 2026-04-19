import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
np.random.seed(42)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1)) 
])

train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

n_samples = 1000
indices = torch.randperm(len(train_dataset))[:n_samples]
X_train = train_dataset.data[indices].float().view(n_samples, -1).to(device)
y_train = train_dataset.targets[indices].to(device)

X_test = test_dataset.data.float().view(len(test_dataset), -1).to(device)
y_test = test_dataset.targets.to(device)

# Standardize
X_mean, X_std = X_train.mean(), X_train.std() + 1e-8
X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

noise_ratio = 0.15 
n_noise = int(noise_ratio * n_samples)
noise_indices = torch.randperm(n_samples)[:n_noise]
y_train[noise_indices] = torch.randint(0, 10, (n_noise,)).to(device)

y_train_oh = torch.nn.functional.one_hot(y_train, 10).float()


def get_rff_features(X, W, b):
    N = W.shape[1]
    return np.sqrt(2.0 / N) * torch.cos(X @ W + b)

def run_experiment(N):
    d = X_train.shape[1]
    W = torch.randn(d, N, device=device) * 0.15 
    b = 2 * np.pi * torch.rand(N, device=device)

    Z_train = get_rff_features(X_train, W, b)
    Z_test = get_rff_features(X_test, W, b)

    # Minimum norm solution
    beta = torch.linalg.lstsq(Z_train, y_train_oh).solution

    train_err = ( (Z_train @ beta).argmax(1) != y_train ).float().mean().item()
    test_err = ( (Z_test @ beta).argmax(1) != y_test ).float().mean().item()

    return train_err, test_err

Ns = np.unique(np.concatenate([
    np.arange(10, 100, 20),      # First Descent area
    np.arange(100, 800, 100),    # Plateau
    np.arange(850, 1150, 25),    # The Peak (Interpolation Threshold)
    np.arange(1200, 4000, 250)   # Second Descent
])).astype(int)

results = {"N": [], "train": [], "test": []}

for N in tqdm(Ns):
    tr, te = run_experiment(N)
    results["N"].append(N)
    results["train"].append(tr)
    results["test"].append(te)


plt.figure(figsize=(10, 6))
plt.plot(results["N"], results["train"], label='Train Error', color='blue', alpha=0.7)
plt.plot(results["N"], results["test"], label='Test Error', color='orange', lw=2)
plt.axvline(x=n_samples, color='red', linestyle='--', label='Interpolation Threshold (N=1000)')

plt.title("Full Double Descent (MNIST, 15% Label Noise)")
plt.xlabel("Features")
plt.ylabel("Classification Error")
plt.xscale('log') 
plt.ylim(0, 1.0)
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.savefig("full_double_descent_MNIST.png")