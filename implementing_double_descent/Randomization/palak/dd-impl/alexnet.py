import argparse
import sys
import pickle
import cv2

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

save_model_path = './alexnet_cifar10.pth'

class AlexNetModel(nn.Module):
    """AlexNet architecture in PyTorch"""
    def __init__(self, num_classes=10):
        super(AlexNetModel, self).__init__()
        
        self.features = nn.Sequential(
            # 1st layer
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # 2nd layer
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # 3rd layer
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # 4th layer
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # 5th layer
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # Add adaptive pooling to ensure consistent output size
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 1)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class AlexNet:
    def __init__(self, dataset='cifar10', learning_rate=0.0001, device=None):
        self.dataset = dataset

        if dataset == "cifar10":
            self.num_classes = 10
        elif dataset == "cifar100":
            self.num_classes = 100
        else:
            self.num_classes = 10

        self.learning_rate = learning_rate
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs_to_overfit = None
        self.history = {
            'train_loss': [], 'train_acc': []
        }
        
        # Build model
        self.model = AlexNetModel(num_classes=self.num_classes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
    def load_data(self, x_train, y_train, batch_size=64):
        """
        Load data into PyTorch DataLoaders.
        
        Args:
            x_train: Training images (numpy array)
            y_train: Training labels (numpy array)
            batch_size: Batch size
        """
        # Convert to tensors
        x_train_tensor = torch.FloatTensor(x_train).permute(0, 3, 1, 2)  # NHWC -> NCHW
        y_train_tensor = torch.LongTensor(y_train)
        
        # Create datasets
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        
        # Create dataloaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        print(f"Dataset loaded:")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Device: {self.device}")
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate on validation set."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self, epochs=20, overfit_threshold=0.01, patience=3, verbose=True):
        """
        Train the model with overfit detection based on training loss.
        
        Args:
            epochs: Maximum number of training epochs
            overfit_threshold: Training loss threshold to detect overfit
            patience: Number of consecutive epochs below threshold to confirm overfit
            verbose: Whether to print training progress
        
        Returns:
            epochs_to_overfit: Number of epochs until overfit
        """
        if not hasattr(self, 'train_loader'):
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Track time to overfit based on training loss
        patience_counter = 0
        epochs_to_overfit = epochs

        print(f'\nStarting training with overfit detection...')
        print(f'Overfit threshold: {overfit_threshold}, Patience: {patience}')
        print('=' * 70)
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            if verbose:
                print(f'Epoch [{epoch+1}/{epochs}] '
                      f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            
            # Check for overfit: training loss below threshold
            if train_loss < overfit_threshold:
                patience_counter += 1
            else:
                patience_counter = 0
            
            # Check if overfitted
            if patience_counter >= patience:
                epochs_to_overfit = epoch + 1 - patience + 1  # First epoch when it went below threshold
                print(f'\nOverfit detected at epoch {epochs_to_overfit} (training loss < {overfit_threshold})')
                break

        print('=' * 70)
        if epochs_to_overfit == epochs:
            print(f'Training completed (did not overfit within {epochs} epochs)')
        else:
            print('Training completed (overfit detected)')
        
        self.epochs_to_overfit = epochs_to_overfit
        return epochs_to_overfit
    
    def save_model(self, path=None):
        """Save the model."""
        if path is None:
            path = save_model_path
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'epochs_to_overfit': self.epochs_to_overfit
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path=None):
        """Load the model."""
        if path is None:
            path = save_model_path
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.epochs_to_overfit = checkpoint.get('epochs_to_overfit', None)
        print(f"Model loaded from {path}")
    
    def plot_history(self):
        """Plot training and validation metrics."""
        if not self.history['train_loss']:
            print("No training history available. Train the model first.")
            return
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot loss
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Train Acc')
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()


def resize_images_to_224(images):
    """Resize images from 32x32 to 224x224 for AlexNet."""
    resized = np.zeros((images.shape[0], 224, 224, 3), dtype=np.float32)
    for i in range(images.shape[0]):
        resized[i] = cv2.resize(images[i], (224, 224))
    return resized


def randomize_labels(labels, corruption_fraction, seed=25292):
    """Randomize a fraction of labels."""
    np.random.seed(seed)
    n_samples = len(labels)
    n_corrupt = int(corruption_fraction * n_samples)
    
    if n_corrupt == 0:
        return labels
    
    corrupted_labels = labels.copy()
    indices = np.random.choice(n_samples, n_corrupt, replace=False)
    
    num_classes = len(np.unique(labels))
    corrupted_labels[indices] = np.random.randint(0, num_classes, n_corrupt)
    
    return corrupted_labels


def parse_args(args):
    parser = argparse.ArgumentParser(description='Script for running AlexNet with PyTorch')

    parser.add_argument('--dataset', help='cifar10 or cifar100, cifar10 is the default', default='cifar10')
    parser.add_argument('--learning-rate', help='learning rate', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--convergence-threshold', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=3)

    return parser.parse_args(args)


def main():
    args = sys.argv[1:]
    args = parse_args(args)

    dataset = args.dataset
    learning_rate = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch_size
    convergence_threshold = args.convergence_threshold
    patience = args.patience
    
    label_corruption_levels = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Load CIFAR-10 dataset
    print('Loading CIFAR-10 dataset...')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=None
    )
    
    # Convert to numpy
    x_train = train_dataset.data.astype(np.float32) / 255.0
    y_train = np.array(train_dataset.targets)
    
    # Resize images to 224x224 for AlexNet
    print('Resizing images to 224x224...')
    x_train_resized = resize_images_to_224(x_train[:5000])  # Use subset for faster training
    y_train_subset = y_train[:5000]
    
    print(f'Training samples: {len(x_train_resized)}')
    
    test_errors = []
    overfit_times = []
    
    for corruption_level in label_corruption_levels:
        print(f"\n{'='*70}")
        print(f"Training with label corruption: {corruption_level:.1%}")
        print(f"{'='*70}")
        
        # Randomize labels
        y_train_corrupted = randomize_labels(y_train_subset.copy(), corruption_level, seed=25292)
        
        # Create model
        model = AlexNet(dataset=dataset, learning_rate=learning_rate)
        
        # Load data
        model.load_data(x_train_resized, y_train_corrupted, batch_size=batch_size)
        
        # Train model (returns epochs to overfit)
        epochs_to_overfit = model.train(
            epochs=epochs,
            overfit_threshold=0.01,
            patience=patience,
            verbose=True
        )
        
        print(f"Epochs to overfit: {epochs_to_overfit}")
        overfit_times.append(epochs_to_overfit)
        
        # Calculate test error (using final training accuracy as proxy)
        final_train_acc = model.history['train_acc'][-1] if model.history['train_acc'] else 0
        test_error = 100 - final_train_acc
        test_errors.append(test_error)
        
        print(f"Final test error (based on training): {test_error:.2f}%")
        
        # Save model
        model.save_model(f'alexnet_cifar10_corruption_{corruption_level:.1f}.pth')
    
    # Plot results
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(label_corruption_levels, test_errors, marker='o')
    plt.title("Test Error vs Label Corruption Level, AlexNet")
    plt.xlabel("Label Corruption Level")
    plt.ylabel("Test Error (%)")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(label_corruption_levels, overfit_times, marker='s', color='green')
    plt.title("Epochs to Overfit vs Label Corruption Level")
    plt.xlabel("Label Corruption Level")
    plt.ylabel("Epochs to Overfit")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("alexnet_analysis.png")
    plt.show()
    
    print("\nAnalysis complete!")
    print(f"Results saved to alexnet_analysis.png")


if __name__ == "__main__":
    main()
