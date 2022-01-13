from datetime import datetime

import cv2
import h5py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

import more_model
from models.defs import transfer_learning, kaggle_model
from utils.consts import IMG_SIZE, font_dict, num_classes, batch_size

#
# os.makedirs("dataset")
#
# for i in range(7):
#     os.makedirs(f"dataset/{i}")

file_path = "./resources/SynthText.h5"
db = h5py.File(file_path, 'r')

im_names = list(db["data"].keys())

images = []
chars = []
fonts = []

print(f"Preprocessing full images to chars images at {datetime.now()}")
for index in range(973):
    # if True:
    #     index = 352
    im = im_names[index]

    img = db['data'][im][:]
    font = db['data'][im].attrs['font']
    font = (list(map(lambda x: font_dict[x], font)))
    txt = b''.join(db['data'][im].attrs['txt'])
    charBB = db['data'][im].attrs['charBB']
    wordBB = db['data'][im].attrs['wordBB']

    pts = np.swapaxes(charBB, 0, 2)

    for idx, char in enumerate(pts):
        char_txt_val = txt[idx]
        char_font_val = font[idx]

        char = np.float32(np.where(char > 0, char, 0))

        # draw_and_show_char_on_image(img, char)

        dst = np.float32([[0, 0], [IMG_SIZE, 0], [IMG_SIZE, IMG_SIZE], [0, IMG_SIZE]])
        mat = cv2.getPerspectiveTransform(char, dst)
        char_image = cv2.warpPerspective(img, mat, (IMG_SIZE, IMG_SIZE))

        images.append(char_image)
        chars.append(char_txt_val)
        fonts.append(char_font_val)

print(f"Finished preprocessing full images to chars images at {datetime.now()}")

# Scaling
images = np.array(images) / 255.0

train_images, test_images, train_chars, test_chars, train_fonts, test_fonts = train_test_split(np.array(images),
                                                                                               np.array(chars),
                                                                                               np.array(fonts),
                                                                                               test_size=0.2)

# Data augment the images
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
])

print(f"images amount {len(images)}")
print(f"train_images amount {len(train_images)}")
print(f"test_images amount {len(test_images)}")
model = tf.keras.Sequential([
    # data_augmentation,  # Will work just at the fit() function. No augmentation for inference stage.
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32, name='image'),
    # more_model.getModel(),
    # transfer_learning.getModel(),
    kaggle_model.getModel(),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy'])

# print(model.summary())

model.fit(train_images,
          train_fonts,
          validation_data=(test_images, test_fonts),
          batch_size=batch_size,
          epochs=24)

