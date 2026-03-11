import tensorflow as tf
from tensorflow.keras import datasets

import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# normalising the data first
x_train = x_train/255.0
x_test = x_test/255.0

#resizing the image because they have used (28, 28) in the paper
x_train = tf.image.resize(x_train, (28, 28))
x_test = tf.image.resize(x_test, (28, 28))

from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPooling2D, Dense, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def inception_block(x, c1, c3, use_bn=True):
  branch1 = Conv2D(c1, (1,1), padding='same')(x)
  if use_bn:
    branch1 = BatchNormalization()(branch1)
  branch1 = ReLU()(branch1)

  branch3 = Conv2D(c3, (3,3), padding='same')(x)
  if use_bn:
    branch3 = BatchNormalization()(branch3)
  branch3 = ReLU()(branch3)

  return Concatenate()([branch1, branch3])


def downsample_block(x, ch3, use_bn=True):
  branch3 = Conv2D(ch3, (3,3), padding='same', strides=(2,2))(x)
  if use_bn:
    branch3 = BatchNormalization()(branch3)
  branch3 = ReLU()(branch3)

  branch1 = MaxPooling2D((3,3), strides=(2,2), padding='same')(x)

  return Concatenate()([branch1, branch3])


def build_model(use_bn=True):
  inputs = tf.keras.Input(shape=(28,28,3))

  # first conv block
  x = Conv2D(96, (3,3), padding='same', strides=(1,1))(inputs)
  if use_bn:
    x = BatchNormalization()(x)
  x = ReLU()(x)

  # first inception block
  x = inception_block(x, 32, 32, use_bn)

  # second inception block
  x = inception_block(x, 32, 48, use_bn)

  # first downsample block
  x = downsample_block(x, 80, use_bn)

  # third inception block
  x = inception_block(x, 112, 48, use_bn)

  # fourth inception block
  x = inception_block(x, 96, 64, use_bn)

  # fifth inception block
  x = inception_block(x, 80, 80, use_bn)

  # sixth inception block
  x = inception_block(x, 48, 96, use_bn)

  # seocnd downsample block
  x = downsample_block(x, 96, use_bn)

  # seventh inception block
  x = inception_block(x, 176, 160, use_bn)

  # eigthth inception block
  x = inception_block(x, 176, 160, use_bn)

  # mean pooling layer
  x = GlobalAveragePooling2D()(x)

  # output layer
  outputs = Dense(10, activation='softmax')(x)

  model = Model(inputs, outputs)

  return model


# with bn
model_bn = build_model(use_bn=True)

model_bn.compile(
    optimizer=tf.keras.optimizers.SGD(
        learning_rate=0.1,
        momentum=0.9
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_bn = model_bn.fit(
    x_train,y_train,
    epochs=50,
    validation_data=(x_test,y_test),
    batch_size=128
)

# without bn
model_no_bn = build_model(use_bn=False)

model_no_bn.compile(
    optimizer=tf.keras.optimizers.SGD(
        learning_rate=0.1,
        momentum=0.9
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_no_bn = model_no_bn.fit(
    x_train,y_train,
    epochs=50,
    validation_data=(x_test,y_test),
    batch_size=128
)   


plt.plot(history_bn.history['accuracy'],label="train(Inception)")
plt.plot(history_bn.history['val_accuracy'],label="test(Inception)")

plt.plot(history_no_bn.history['accuracy'],label="train(Inception w/o BN)")
plt.plot(history_no_bn.history['val_accuracy'],label="test(Inception w/o BN)")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("accuracy_plot.png")