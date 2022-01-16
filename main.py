from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

from models.defs import kaggle_model
from utils.consts import IMG_SIZE, num_classes, batch_size
from utils.data_manipulators import preprocess_h5_dataset

file_path = "./resources/SynthText.h5"

images, chars, fonts, _, _ = preprocess_h5_dataset(file_path)

# Scaling
images = np.array(images) / 255.0
chars = np.array(chars) / 127.0

train_images, test_images, train_chars, test_chars, train_fonts, test_fonts = train_test_split(np.array(images),
                                                                                               np.array(chars),
                                                                                               np.array(fonts),
                                                                                               test_size=0.2)
# Data augment the images
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.1),
])

print(f"images amount {len(images)}")
print(f"train_images amount {len(train_images)}")
print(f"test_images amount {len(test_images)}")

# Functional with multi inputs - image and char value.
image_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32, name='image')
x = data_augmentation(image_input)
x = kaggle_model.getModel()(x)
# x = layers.Flatten()(x)

char_input = layers.Input(shape=(1,), dtype=tf.float32, name='char')

combined = layers.concatenate([x, char_input])
combined = layers.Dense(128, activation='relu')(combined)
combined = layers.Dense(64, activation='relu')(combined)
outputs = layers.Dense(num_classes, activation='softmax')(combined)

model = tf.keras.Model(inputs=[image_input, char_input], outputs=outputs)

# # Worked Sequential
# model = tf.keras.Sequential([
#     # data_augmentation,  # Will work just at the fit() function. No augmentation for inference stage.
#     layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32, name='image', ),
#     # more_model.getModel(),
#     # transfer_learning.getModel(),
#     kaggle_model.getModel(),
#     layers.Dense(num_classes, activation='softmax')
# ])

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy'])


# my_model.build()
dot_img_file = r'C:\devl\study\ComputerVisionProject\models\images\main_model.png'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

# print(model.summary())
start_time = datetime.now()
print(f"Starting training at {start_time}")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`")

inputs = [train_images, train_chars]

model.fit(inputs,
          train_fonts,
          validation_data=([test_images, test_chars], test_fonts),
          batch_size=batch_size,
          epochs=25)
print(f"Finished training at {(datetime.now() - start_time).total_seconds()}")

model.save("saved_model.h5")
