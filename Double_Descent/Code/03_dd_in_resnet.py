import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# dataset
class CIFAR10Dataset:
    def __init__(self, subset=2000, noise_level=0.2):
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


# resnet
class ResNetModel:
    def __init__(self, k):
        self.k = k

    def res_block(self, x, filters):
        shortcut = x

        x = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)

        if shortcut.shape[-1] != filters:
            shortcut = tf.keras.layers.Conv2D(filters, 1, padding='same')(shortcut)

        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.Activation('relu')(x)
        return x

    def build(self):
        inputs = tf.keras.Input(shape=(32, 32, 3))

        x = tf.keras.layers.Conv2D(self.k, 3, padding='same', activation='relu')(inputs)

        x = self.res_block(x, self.k)
        x = tf.keras.layers.MaxPooling2D()(x)

        x = tf.keras.layers.Conv2D(2*self.k, 3, padding='same', activation='relu')(x)
        x = self.res_block(x, 2*self.k)
        x = tf.keras.layers.MaxPooling2D()(x)

        x = tf.keras.layers.Conv2D(4*self.k, 3, padding='same', activation='relu')(x)
        x = self.res_block(x, 4*self.k)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

        return tf.keras.Model(inputs, outputs)


# training
class Trainer:
    def __init__(self, epsilon=0.00001, max_epochs=1000, batch_size=128):
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
            train_acc = history.history['accuracy'][0]

            if train_loss < min_train_loss:
                min_train_loss = train_loss
                best_weights = model.get_weights()

            print(f"Epoch {epoch+1} | Train Loss={train_loss:.4f}")

            if min_train_loss <= self.epsilon:
                print(f"Stopping at epoch {epoch+1} (min loss reached)")
                break

        model.set_weights(best_weights)

        train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

        return train_loss, test_loss, train_acc, test_acc


# experiment
class Experiment:
    def __init__(self, widths):
        self.widths = widths
        self.trainer = Trainer()

    def run(self, x_train, y_train, x_test, y_test):
        tl, vl, ta, va = [], [], [], []

        for k in self.widths:
            print(f"\n=== k={k} ===")
            model = ResNetModel(k).build()
            a, b, c, d = self.trainer.train(model, x_train, y_train, x_test, y_test)

            tl.append(a)
            vl.append(b)
            ta.append(c)
            va.append(d)

        return tl, vl, ta, va


# plotting
def plot(widths, tl, vl, ta, va):
    plt.figure()
    plt.plot(widths, tl, 'o-', label='Train Loss')
    plt.plot(widths, vl, 'o-', label='Test Loss')
    plt.legend(); plt.grid(); plt.title("ResNet Loss")
    plt.savefig("double_descent_resnet_loss01.png")
    plt.show()

    plt.figure()
    plt.plot(widths, ta, 'o-', label='Train Acc')
    plt.plot(widths, va, 'o-', label='Test Acc')
    plt.legend(); plt.grid(); plt.title("ResNet Accuracy")
    plt.savefig("double_descent_resnet_acc.png")
    plt.show()



dataset = CIFAR10Dataset()
x_train, y_train, x_test, y_test = dataset.load_data()


widths = [4, 6, 8, 10, 12, 16, 32, 64, 112]

exp = Experiment(widths)
tl, vl, ta, va = exp.run(x_train, y_train, x_test, y_test)

plot(widths, tl, vl, ta, va)