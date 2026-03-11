import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np
import random
import matplotlib.pyplot as plt

############################################
# Seed for reproducibility
############################################

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################################
# Load CIFAR10 raw arrays
############################################

dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True
)

X = dataset.data
y = np.array(dataset.targets)

############################################
# Dataset modification functions
############################################

def true_labels(X,y):
    return X.copy(), y.copy()


def partially_corrupt_labels(X,y,p=0.3):

    y_new = y.copy()

    mask = np.random.rand(len(y)) < p
    y_new[mask] = np.random.randint(0,10,mask.sum())

    return X.copy(), y_new


def random_labels(X,y):

    y_new = np.random.randint(0,10,len(y))
    return X.copy(), y_new


def shuffled_pixels(X,y):

    X_flat = X.reshape(len(X),-1)

    perm = np.random.permutation(X_flat.shape[1])
    X_flat = X_flat[:,perm]

    X_new = X_flat.reshape(X.shape)

    return X_new, y.copy()


def random_pixels(X,y):

    X_new = np.zeros_like(X)

    for i in range(len(X)):

        flat = X[i].flatten()
        perm = np.random.permutation(len(flat))

        X_new[i] = flat[perm].reshape(32,32,3)

    return X_new, y.copy()


def gaussian_pixels(X,y):

    mean = X.mean()
    std = X.std()

    X_new = np.random.normal(mean,std,X.shape)
    X_new = np.clip(X_new,0,255).astype(np.uint8)

    return X_new, y.copy()

############################################
# Dataset wrapper (avoids huge RAM usage)
############################################

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(299),
    transforms.ToTensor()
])


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self,X,y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):

        img = transform(self.X[idx])
        label = self.y[idx]

        return img, label


############################################
# Model: Inception V3
############################################

def create_model():

    model = models.inception_v3(
        weights=None,
        aux_logits=False,
        init_weights=False
    )

    model.fc = nn.Linear(model.fc.in_features,10)

    return model.to(device)


############################################
# Training function
############################################

def train_model(X,y,steps=25000):

    dataset = CustomDataset(X,y)

    g = torch.Generator()
    g.manual_seed(42)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        generator=g
    )

    model = create_model()

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9
    )
    optimizer.step()
    torch.cuda.empty_cache()
    criterion = nn.CrossEntropyLoss()

    losses = []
    step = 0

    while step < steps:

        for imgs,labels in loader:

            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(imgs)

            loss = criterion(outputs,labels)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            step += 1

            if step % 1000 == 0:
                print("step:",step,"loss:",loss.item())

            if step >= steps:
                break

    return losses


############################################
# Run experiments
############################################

experiments = {
    "true_labels": true_labels,
    "partial_labels": lambda X,y: partially_corrupt_labels(X,y,0.3),
    "random_labels": random_labels,
    "shuffled_pixels": shuffled_pixels,
    "random_pixels": random_pixels,
    "gaussian_pixels": gaussian_pixels
}

results = {}

for name,func in experiments.items():

    print("\nRunning:",name)

    X_mod,y_mod = func(X,y)

    losses = train_model(X_mod,y_mod)

    results[name] = losses


############################################
# Plot results
############################################

plt.figure(figsize=(8,6))

for name,losses in results.items():

    steps = np.arange(len(losses))/1000
    plt.plot(steps,losses,label=name)

plt.xlabel("Thousands of steps")
plt.ylabel("Training loss")

plt.xticks(np.arange(0,26,5))

plt.title("Fitting Random Labels and Random Pixels on CIFAR10")

plt.legend()

plt.show()

plt.savefig("random_experiments.png")