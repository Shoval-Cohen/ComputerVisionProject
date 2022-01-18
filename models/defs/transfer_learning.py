import tensorflow as tf
from keras import layers
from tensorflow import keras

from utils.consts import IMG_SIZE

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
])
