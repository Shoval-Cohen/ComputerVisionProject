import os
from datetime import datetime

import cv2
import h5py
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers

from utils.consts import IMG_SIZE, font_dict

tf.compat.v1.enable_eager_execution()

AUTOTUNE = tf.data.AUTOTUNE


def preprocess_h5_dataset(file_path):
    db = h5py.File(file_path, 'r')

    im_names = list(db["data"].keys())

    images = []
    chars = []
    words = []
    fonts = []
    big_img_names = []

    start_time = datetime.now()
    print(f"Preprocessing full images to chars images at {start_time}")
    for index in range(973):
        # if True:
        #     index = 352
        im = im_names[index]

        img = db['data'][im][:]
        font = db['data'][im].attrs['font']
        font = (list(map(lambda x: font_dict[x], font)))
        fonts.extend(font)
        txt = db['data'][im].attrs['txt']
        words.extend(txt)
        images_chars = b''.join(txt)
        chars.extend(images_chars)
        charBB = db['data'][im].attrs['charBB']
        # wordBB = db['data'][im].attrs['wordBB']

        char_pts = np.swapaxes(charBB, 0, 2)

        for char in char_pts:
            char = np.float32(np.where(char > 0, char, 0))

            dst = np.float32([[0, 0], [IMG_SIZE, 0], [IMG_SIZE, IMG_SIZE], [0, IMG_SIZE]])
            mat = cv2.getPerspectiveTransform(char, dst)
            char_image = cv2.warpPerspective(img, mat, (IMG_SIZE, IMG_SIZE))
            # if (im == "aquarium_31.jpg_0"):
            #     cv2.imshow("NAme", char_image)
            #     draw_and_show_char_on_image(img, char)
            big_img_names.append(im)
            images.append(char_image)

    print(
        f"Finished preprocessing full images to chars images at {datetime.now()}.",
        f"Took: {(datetime.now() - start_time).total_seconds()}"
    )

    return images, chars, fonts, words, big_img_names


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


def split_and_prepare(ds, validation_rate=0.8, multi_input=True):
    ds = ds.map(decode_char_font_and_image)

    print("Decoded!")

    image_count = len(ds)
    train_size = int(image_count * validation_rate)

    train_ds = prepare(ds.take(train_size), multi_input=multi_input, shuffle=True, augment=True)
    val_ds = prepare(ds.skip(train_size), multi_input=multi_input)

    return train_ds, val_ds


def draw_and_show_char_on_image(img, char_img):
    rounded_char = char_img.copy()
    rounded_char = np.asarray(rounded_char, np.int32)
    orig = cv2.polylines(img.copy(), np.asarray([rounded_char]), True, color=(0, 0, 255))
    cv2.imshow("Image with char", orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ds = tf.data.Dataset.list_files("../dataset/*/*")
# _, _ = split_and_prepare(ds)


preprocess_h5_dataset(r"../resources/SynthText.h5")
