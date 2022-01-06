import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

data_dir = "./dataset"

img_height = 100,
img_width = 100,
num_classes = 7


def display_9_images_from_dataset(dataset):
    plt.figure(figsize=(13, 13))
    subplot = 331
    for i, (image, char, label) in enumerate(dataset):
        plt.subplot(subplot)
        plt.axis('off')
        plt.imshow(image.numpy().astype(np.uint8))
        plt.title(f'Char = {chr(char.numpy())}, font={label.numpy()}', fontsize=12)
        subplot += 1
        if i == 8:
            break
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()


def decode_img(filename):
    bits = tf.io.read_file(filename)
    image = tf.io.decode_png(bits)
    return image


def decode_char_font_and_image(file_path):
    font_num = int(tf.strings.split(file_path, os.path.sep)[1])
    char_val = int(tf.strings.split(file_path, '_')[-3])
    img = decode_img(file_path)
    return img, char_val, font_num


# display_9_images_from_dataset(train_ds)

IMG_SIZE = 180

resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMG_SIZE, IMG_SIZE),
    layers.Rescaling(1. / 255)
])

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])

batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE


def prepare(ds, shuffle=False, augment=False):
    # Resize and rescale all datasets.
    ds = ds.map(lambda img, cr, fnt: (resize_and_rescale(img), cr, fnt),
                num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1000)

    # Batch all datasets.
    ds = ds.batch(batch_size)

    # Use data augmentation only on the training set.
    if augment:
        ds = ds.map(lambda img, cr, fnt: (data_augmentation(img, training=True), cr, fnt),
                    num_parallel_calls=AUTOTUNE)

    # Use buffered prefetching on all datasets.
    return ds.prefetch(buffer_size=AUTOTUNE)


ds = tf.data.Dataset.list_files("dataset/*/*").map(decode_char_font_and_image)

image_count = len(ds)
validation_rate = 0.8
train_size = int(image_count * validation_rate)

train_ds = ds.take(train_size)
val_ds = ds.skip(train_size)

train_ds = prepare(train_ds, shuffle=True, augment=True)
val_ds = prepare(val_ds)

for img, char_val, font_num in train_ds.take(5).as_numpy_iterator():
    print(f"char_val: {char_val}")
    # f" = {chr(char_val)}")
    print(f"font_num: {font_num}")
    print(f"img: {img}")

cnn = tf.keras.Sequential([
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

image_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='image')
x = cnn(image_input)
x = layers.Flatten()(x)

char_input = layers.Input(shape=(1,), name='char')

combined = layers.concatenate([x, char_input])
combined = layers.Dense(1024)(combined)
combined = tf.nn.relu(combined)
combined = layers.Dense(128)(combined)
combined = tf.nn.relu(combined)
outputs = layers.Dense(num_classes)(combined)
outputs = tf.nn.softmax(outputs)

model = tf.keras.Model(inputs=[image_input, char_input], outputs=outputs)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(
    {"image": train_ds[0], "char": train_ds[1]},
    train_ds[2],
    epoch=2,
    batch_size=32
)

print(train_ds)
# model.fit(train_ds)


# train_ds = tf.keras.utils.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.2,
#   subset="training",
#   seed=123,
#   image_size=(100, 100),
#   # batch_size=batch_size
#   )
#
# for image_batch, labels_batch in train_ds:
#   print(image_batch.shape)
#   print(labels_batch.shape)
#   break
#
# # for element in train_ds[:10]:
# #   print(element)
#
# val_ds = tf.keras.utils.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.2,
#   subset="validation",
#   seed=123,
#   image_size=(100, 100),
#   # batch_size=batch_size
#   )
