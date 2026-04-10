import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class CIFAR10Dataset:
    def __init__(self, subset=4000, noise_level=0.2):
        self.subset = subset
        self.noise_level = noise_level

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        x_train = x_train / 255.0
        x_test = x_test / 255.0

        x_train = x_train[:self.subset]
        y_train = y_train[:self.subset]

        y_train = y_train.flatten()
        y_test = y_test.flatten()

        y_train = self.add_label_noise(y_train)

        return x_train, y_train, x_test, y_test

    def add_label_noise(self, y):
        y_noisy = y.copy()
        mask = np.random.rand(len(y)) < self.noise_level
        y_noisy[mask] = np.random.randint(0, 10, np.sum(mask))
        return y_noisy



class ResNetModel:
    def __init__(self, k, initializer):
        self.k = k
        self.initializer = initializer

    def res_block(self, x, filters):
        shortcut = x

        x = tf.keras.layers.Conv2D(
            filters, 3, padding='same', activation='relu',
            kernel_initializer=self.initializer
        )(x)

        x = tf.keras.layers.Conv2D(
            filters, 3, padding='same',
            kernel_initializer=self.initializer
        )(x)

        if shortcut.shape[-1] != filters:
            shortcut = tf.keras.layers.Conv2D(
                filters, 1, padding='same',
                kernel_initializer=self.initializer
            )(shortcut)

        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.Activation('relu')(x)

        return x

    def build(self):
        inputs = tf.keras.Input(shape=(32, 32, 3))

        x = tf.keras.layers.Conv2D(
            self.k, 3, padding='same', activation='relu',
            kernel_initializer=self.initializer
        )(inputs)

        x = self.res_block(x, self.k)
        x = tf.keras.layers.MaxPooling2D()(x)

        x = tf.keras.layers.Conv2D(
            2 * self.k, 3, padding='same', activation='relu',
            kernel_initializer=self.initializer
        )(x)

        x = self.res_block(x, 2 * self.k)
        x = tf.keras.layers.MaxPooling2D()(x)

        x = tf.keras.layers.Conv2D(
            4 * self.k, 3, padding='same', activation='relu',
            kernel_initializer=self.initializer
        )(x)

        x = self.res_block(x, 4 * self.k)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        outputs = tf.keras.layers.Dense(
            10, activation='softmax',
            kernel_initializer=self.initializer
        )(x)

        return tf.keras.Model(inputs, outputs)



class Trainer:
    def __init__(self, epsilon=1e-4, max_epochs=5000, batch_size=128):
        self.epsilon = epsilon
        self.max_epochs = max_epochs
        self.batch_size = batch_size

    def train(self, model, x_train, y_train, x_test, y_test):
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        best_weights = None
        min_train_loss = float('inf')

        for epoch in range(self.max_epochs):
            history = model.fit(
                x_train, y_train,
                epochs=1,
                batch_size=self.batch_size,
                verbose=1
            )

            train_loss = history.history['loss'][0]

            print(f"Epoch {epoch+1} | Loss={train_loss:.5f} | Weight Norm={np.linalg.norm(model.get_weights()[0]):.4f}")

            if train_loss < min_train_loss:
                min_train_loss = train_loss
                best_weights = model.get_weights()

            if min_train_loss <= self.epsilon:
                print(f"Early stop at epoch {epoch+1}")
                break

        model.set_weights(best_weights)

        train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

        return train_loss, test_loss, train_acc, test_acc



class Experiment:
    def __init__(self, widths, initializers):
        self.widths = widths
        self.initializers = initializers
        self.trainer = Trainer()

    def run(self, x_train, y_train, x_test, y_test):
        results = {}

        for name, init in self.initializers.items():
            print(f"\n Initializer: {name} ")

            tl, vl, ta, va = [], [], [], []

            for k in self.widths:
                print(f"\n Width k={k} ")

                model = ResNetModel(k, init).build()

                train_loss, test_loss, train_acc, test_acc = \
                    self.trainer.train(model, x_train, y_train, x_test, y_test)

                tl.append(train_loss)
                vl.append(test_loss)
                ta.append(train_acc)
                va.append(test_acc)

            results[name] = (tl, vl, ta, va)

        return results


# =========================
# Plot
# =========================
def plot(widths, results):
    plt.figure()

    for name, (tl, vl, _, _) in results.items():
        plt.plot(widths, tl, 'o--', label=f"{name} - Train")
        plt.plot(widths, vl, 'o-', label=f"{name} - Test")

    plt.title("Train & Test Loss vs Width")
    plt.xlabel("Width (k)")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig("tenkresnet_loss.png")
    plt.show()



dataset = CIFAR10Dataset(subset=10000, noise_level=0.2)
x_train, y_train, x_test, y_test = dataset.load_data()

widths = [4, 6, 8, 10, 12, 16, 32, 64, 112]

initializers = {
    # "glorot_uniform": tf.keras.initializers.GlorotUniform(),
    # "he_normal": tf.keras.initializers.HeNormal(),
    "normal": tf.keras.initializers.RandomNormal(stddev=0.01)
}

exp = Experiment(widths, initializers)
results = exp.run(x_train, y_train, x_test, y_test)

plot(widths, results)