import datetime
import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

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

IMG_SIZE = 150

resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMG_SIZE, IMG_SIZE),
    layers.Rescaling(1. / 255)
])

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])

batch_size = 64
AUTOTUNE = tf.data.AUTOTUNE


def prepare(ds, shuffle=False, augment=False, multi_input=False):
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

    if multi_input:
        ds = ds.map(lambda img, cr, fnt: ({'image': img, 'char': cr}, fnt))
    else:
        ds = ds.map(lambda img, cr, fnt: (img, fnt))
    # Use buffered prefetching on all datasets.
    return ds.prefetch(buffer_size=AUTOTUNE)


ds = tf.data.Dataset.list_files("dataset/*/*").map(decode_char_font_and_image)

image_count = len(ds)
validation_rate = 0.8
train_size = int(image_count * validation_rate)

train_ds = ds.take(train_size)
val_ds = ds.skip(train_size)

print(f"Trying to prepare {datetime.datetime.now()}")
train_ds = prepare(train_ds, multi_input=True, shuffle=True, augment=True)
val_ds = prepare(val_ds, multi_input=True)
print(f"Finished preparation {datetime.datetime.now()}")

# for img, font_num in train_img_ds.take(5).as_numpy_iterator():
#     # print(f"char_val: {x[1]}")
#     # # f" = {chr(char_val)}")
#     print(f"font_num: {font_num}")
#     print(f"img: {img}")


base_model = tf.keras.applications.Xception(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)

base_model.trainable = False

transfer_learning_model = tf.keras.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    # resize_and_rescale,
    # data_augmentation,
    # PreTrained model
    base_model,
    layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(127, activation='relu'),
    # tf.keras.layers.Dense(num_classes)
])

cnn = tf.keras.Sequential([
    # hub.KerasLayer(model_handle, trainable=True),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(num_classes,
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001))
])

image_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32, name='image')
x = transfer_learning_model(image_input)
# x = layers.Flatten()(x)

char_input = layers.Input(shape=(1,), dtype=tf.float32, name='char')
#
# combined = layers.concatenate([x, char_input])
# combined = layers.Dense(64,activation='relu')(combined)
outputs = layers.Dense(num_classes, activation='softmax')(x)


model = tf.keras.Model(inputs=[image_input, char_input], outputs=outputs)
#
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.1,
    decay_steps=10000,
    decay_rate=0.9)

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy'])

dot_img_file = 'model.png'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

print(transfer_learning_model.summary())

print("Let's go!")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

model.save("saved_model.h5")
