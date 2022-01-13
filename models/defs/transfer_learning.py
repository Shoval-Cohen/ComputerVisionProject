import os

import tensorflow as tf
from keras import layers
from tensorflow import keras

from utils.consts import IMG_SIZE

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print("TensorFlow version:", tf.__version__)

num_classes = 7

base_model = keras.applications.Xception(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False)  # Do not include the ImageNet classifier at the top.

base_model.trainable = False

my_model = tf.keras.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    # PreTrained model
    base_model,
    layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.Dense(num_classes)
])


# dot_img_file = 'models/images/transfer_learning.model'
# tf.keras.utils.plot_model(my_model, to_file=dot_img_file, show_shapes=True)


def getModel():
    my_model.compile(optimizer='adam',
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])
    return my_model
