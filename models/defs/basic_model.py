import tensorflow as tf
from tensorflow.keras import Sequential, layers

from utils.consts import IMG_SIZE, num_classes

basic_model = Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32, name='image', ),

    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    # Flatten the results to feed into a DNN
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    # Classification layer
    layers.Dense(num_classes, activation='softmax')
])
