import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models, optimizers, utils, datasets
import tensorflow as tf
from cifar10_utils import randomize_labels


class MPL1_512:
    def __init__(self, input_shape=(32, 32, 3), num_classes=10, learning_rate=0.0005):
        """
        Initialize the Multi-Layer Perceptron with 1 hidden layer of 512 units.

        Args:
            input_shape: Shape of input images (default: (32, 32, 3) for CIFAR-10)
            num_classes: Number of output classes (default: 10)
            learning_rate: Learning rate for Adam optimizer (default: 0.0005)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        self.epochs_to_overfit = None

        self._build_model()

    def init_weights(self, layer):
        if isinstance(layer, layers.Dense):
            initializer = tf.keras.initializers.GlorotUniform(seed=25292)
            layer.kernel_initializer = initializer
            layer.bias_initializer = initializer

    def _build_model(self):
        """Build the neural network architecture."""
        input_layer = layers.Input(self.input_shape)

        x = layers.Flatten()(input_layer)
        x = layers.Dense(512, activation="relu")(x)
        output_layer = layers.Dense(self.num_classes, activation="softmax")(x)

        self.model = models.Model(input_layer, output_layer)

        opt = optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(
            loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
        )

    def summary(self):
        """Display model architecture summary."""
        return self.model.summary()

    def train(
        self,
        x_train,
        y_train,
        x_test=None,
        y_test=None,
        epochs=500,
        batch_size=128,
        verbose=1,
        overfit_threshold=0.01,
        patience=3,
        label_corruption_level=0.0,
    ):
        """
        Train the model with overfit detection based on training loss.

        Args:
            x_train: Training data
            y_train: Training labels
            x_test: Test/validation data for per-epoch loss tracking (default: None)
            y_test: Test/validation labels for per-epoch loss tracking (default: None)
            epochs: Maximum number of training epochs (default: 200)
            batch_size: Batch size (default: 128)
            verbose: Verbosity mode (default: 1)
            overfit_threshold: Training loss threshold to detect overfit (default: 0.01)
            patience: Number of consecutive epochs below threshold to confirm overfit (default: 3)

        Returns:
            Training history, final test error, and epochs to overfit
        """

        # Track time to overfit based on training loss
        patience_counter = 0
        epochs_to_overfit = epochs  # Default to max epochs if no overfit detected

        # Initialize history tracking for epoch-wise plotting
        all_history = {"loss": [], "accuracy": [], "test_loss": []}

        for epoch in range(epochs):
            history = self.model.fit(
                x_train, y_train, batch_size=batch_size, epochs=1, verbose=verbose
            )

            # Get current training loss
            current_train_loss = history.history["loss"][0]

            # Evaluate test loss at each epoch if test data is provided
            current_test_loss = np.nan
            if x_test is not None and y_test is not None:
                current_test_loss, _ = self.model.evaluate(x_test, y_test, verbose=0)

            # Store history
            for key in history.history.keys():
                if key not in all_history:
                    all_history[key] = []
                all_history[key].append(history.history[key][0])
            all_history["test_loss"].append(current_test_loss)

            # Check for overfit: training loss below threshold
            if current_train_loss < overfit_threshold:
                patience_counter += 1
            else:
                patience_counter = 0

            # Check if overfit detected
            if patience_counter >= patience:
                epochs_to_overfit = (
                    epoch + 1 - patience + 1
                )  # First epoch when it went below threshold
                if verbose:
                    print(
                        f"\nOverfit detected at epoch {epochs_to_overfit} (training loss < {overfit_threshold})"
                    )
                break

        # Plot train/test loss against epochs
        epoch_indices = range(1, len(all_history["loss"]) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epoch_indices, all_history["loss"], linestyle=":", label="Train Loss")
        plt.plot(
            epoch_indices, all_history["test_loss"], color="orange", label="Test Loss"
        )
        plt.title("Train vs Test Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"MPL1_512_train_test_loss_{k}.png")
        plt.close()

        self.epochs_to_overfit = epochs_to_overfit

        final_test_error = np.nan
        if x_test is not None and y_test is not None:
            _, final_test_acc = self.model.evaluate(x_test, y_test, verbose=0)
            final_test_error = 1 - final_test_acc

        if verbose and epochs_to_overfit == epochs:
            print(
                f"\nDid not overfit within {epochs} epochs (training loss did not stay below {overfit_threshold})"
            )

        self.history = all_history
        return self.history, final_test_error, epochs_to_overfit

    def evaluate(self, x_test, y_test):
        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=0)
        test_error = 1 - test_acc
        print(
            f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test Error: {test_error:.4f}"
        )
        return test_loss, test_acc, test_error


if __name__ == "__main__":
    # Load and preprocess CIFAR-10 data
    print("starting here")
    NUM_CLASSES = 10
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train = utils.to_categorical(y_train, NUM_CLASSES)
    y_test = utils.to_categorical(y_test, NUM_CLASSES)

    p = [0.0]
    seed = 25292
    test_errors = []
    time_to_overfit = []
    for k in p:
        """p is the fraction that stays the same and 1-p is the fraction that gets randomized"""
        print(f"Randomization level: {k}")
        y_train_randomized = randomize_labels(y_train.copy(), 1 - k, seed)
        y_test_randomized = randomize_labels(y_test.copy(), 1 - k, seed)
        model = MPL1_512(num_classes=NUM_CLASSES)

        _, final_test_error, epochs_to_overfit_val = model.train(
            x_train,
            y_train_randomized,
            x_test,
            y_test_randomized,
            epochs=100,
            batch_size=128,
            label_corruption_level=1 - k,
        )
        test_errors.append(final_test_error)
        time_to_overfit.append(epochs_to_overfit_val)
        print(f"  Final test error: {final_test_error:.4f}")
        print(f"  Epochs to overfit: {epochs_to_overfit_val}")

    # Plot test error vs randomization
    plt.figure(figsize=(12, 5))

    