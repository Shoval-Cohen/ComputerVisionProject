from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard

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

# # Functional with multi inputs - image and char value.
# image_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32, name='image')
# x = data_augmentation(image_input)
# x = kaggle_model.getModel()(x)
# # x = layers.Flatten()(x)
#
# char_input = layers.Input(shape=(1,), dtype=tf.float32, name='char')
#
# combined = layers.concatenate([x, char_input])
# combined = layers.Dense(128, activation='relu')(combined)
# combined = layers.Dropout(0.5)(combined)
# combined = layers.Dense(64, activation='relu')(combined)
# outputs = layers.Dense(num_classes, activation='softmax')(combined)
#
# model = tf.keras.Model(inputs=[image_input, char_input], outputs=outputs)

# Worked Sequential
model = tf.keras.Sequential([
    # data_augmentation,  # Will work just at the fit() function. No augmentation for inference stage.
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32, name='image', ),
    # more_model.getModel(),
    # transfer_learning.getModel(),
    # This is the first convolution
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(2, 2),
    # The second convolution
    # layers.Conv2D(64, (3, 3), activation='relu'),
    # layers.MaxPooling2D(2, 2),
    # The third convolution
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    # The fourth convolution
    # layers.Conv2D(128, (3, 3), activation='relu'),
    # layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    layers.Flatten(),
    layers.Dropout(0.5),
    # 512 neuron hidden layer
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True),
model.compile(optimizer='Adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# my_model.build()
dot_img_file = r'C:\devl\study\ComputerVisionProject\models\images\main_model.png'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

# print(model.summary())
start_time = datetime.now()
print(f"Starting training at {start_time}")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`")

log_folder = "logs/Basic/Adam/" + datetime.now().strftime("%Y%m%d-%H%M%S")
callbacks = [TensorBoard(log_dir=log_folder,
                         histogram_freq=1,
                         write_graph=True,
                         write_images=True,
                         update_freq='epoch',
                         profile_batch=2,
                         embeddings_freq=1)]

# model.fit([train_images, train_chars],
model.fit(train_images,
          train_fonts,
          # validation_data=(test_images, test_fonts),
          validation_data=(test_images, test_fonts),
          batch_size=batch_size,
          epochs=25,
          callbacks=callbacks)

print(f"Finished training at {(datetime.now() - start_time).total_seconds()}")

model.save("saved_model.h5")
