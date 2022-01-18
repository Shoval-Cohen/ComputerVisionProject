import tensorflow as tf
from tensorflow.keras import layers

from utils.consts import IMG_SIZE, num_classes

# Functional with multi inputs - image and char value.
image_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32, name='image')
conv_results = layers.Conv2D(64, (3, 3), activation='relu')(image_input)
conv_results = layers.MaxPooling2D(2, 2)(conv_results)

conv_results = layers.Conv2D(128, (3, 3), activation='relu')(conv_results)
conv_results = layers.MaxPooling2D(2, 2)(conv_results)

# Flatten the results to feed into a DNN
conv_results = layers.Flatten()(conv_results)
conv_results = layers.Dropout(0.5)(conv_results)

char_input = layers.Input(shape=(1,), dtype=tf.float32, name='char')

combined = layers.concatenate([conv_results, char_input])

combined = layers.Dense(128, activation='relu')(combined)
combined = layers.Dropout(0.5)(combined)
combined = layers.Dense(64, activation='relu')(combined)
outputs = layers.Dense(num_classes, activation='softmax')(combined)

multi_input_model = tf.keras.Model(inputs=[image_input, char_input], outputs=outputs)
