import os

import tensorflow as tf
from keras import layers
from tensorflow import keras

from utils.consts import IMG_SIZE

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

base_model = keras.applications.ResNet50(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False
)  # Do not include the ImageNet classifier at the top.

base_model.trainable = False

transfer_learning_model = tf.keras.Sequential([
    # PreTrained model
    base_model,
    layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.Dropout(0.5)
    # tf.keras.layers.Dense(num_classes)
])
