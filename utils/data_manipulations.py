from datetime import datetime

import cv2
import h5py
import numpy as np

from utils.consts import IMG_SIZE, font_dict


def preprocess_h5_dataset(file_path, is_training=True):
    # init
    images = []
    chars = []
    words = []
    fonts = []
    txt_img_names = []

    db = h5py.File(file_path, 'r')

    im_names = list(db["data"].keys())

    start_time = datetime.now()
    print(f"Preprocessing full images to chars images at {start_time}")
    for index in range(len(list(db["data"].keys()))):

        image_name = im_names[index]

        img = db['data'][image_name][:]

        txt = db['data'][image_name].attrs['txt']
        words.extend(txt)
        images_chars = b''.join(txt)
        chars.extend(images_chars)
        char_BB = db['data'][image_name].attrs['charBB']

        char_pts = np.swapaxes(char_BB, 0, 2)

        for char in char_pts:
            char = np.float32(np.where(char > 0, char, 0))

            dst = np.float32([[0, 0], [IMG_SIZE, 0], [IMG_SIZE, IMG_SIZE], [0, IMG_SIZE]])
            mat = cv2.getPerspectiveTransform(char, dst)
            char_image = cv2.warpPerspective(img, mat, (IMG_SIZE, IMG_SIZE))

            txt_img_names.append(image_name)
            images.append(char_image)

        if is_training:
            font = db['data'][image_name].attrs['font']
            font = list(map(lambda x: font_dict[x], font))
            fonts.extend(font)

    print(
        f"Finished preprocessing full images to chars images at {datetime.now()}.",
        f"Took: {(datetime.now() - start_time).total_seconds()}"
    )

    return images, chars, fonts, words, txt_img_names
