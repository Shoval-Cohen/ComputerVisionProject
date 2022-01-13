import os

import cv2
import h5py
import numpy as np

from utils.consts import IMG_SIZE

os.makedirs("dataset")

for i in range(7):
    os.makedirs(f"dataset/{i}")

file_path = "./resources/SynthText.h5"
db = h5py.File(file_path, 'r')

im_names = list(db["data"].keys())
font_dict = {
    b'Raleway': 0,
    b'Open Sans': 1,
    b'Roboto': 2,
    b'Ubuntu Mono': 3,
    b'Michroma': 4,
    b'Alex Brush': 5,
    b'Russo One': 6
}

for index in range(973):
    # if True:
    #     index = 352

    im = im_names[index]
    print(f"working on image {index} with name {im}")

    img = db['data'][im][:]
    font = db['data'][im].attrs['font']
    font = (list(map(lambda x: font_dict[x], font)))
    txt = b''.join(db['data'][im].attrs['txt'])
    charBB = db['data'][im].attrs['charBB']
    wordBB = db['data'][im].attrs['wordBB']

    pts = np.swapaxes(charBB, 0, 2)

    # np.where(font ,font_dict[str(font)], font)
    for idx, char in enumerate(pts):
        char_txt_val = txt[idx]
        char_font_val = font[idx]

        char = np.float32(np.where(char > 0, char, 0))
        # rounded_char = char.copy()
        # rounded_char = np.asarray(char, np.int32)
        # orig = cv2.polylines(img.copy(), np.asarray([rounded_char]), True, color=(0, 0, 255))

        dst = np.float32([[0, 0], [IMG_SIZE, 0], [IMG_SIZE, IMG_SIZE], [0, IMG_SIZE]])

        mat = cv2.getPerspectiveTransform(char, dst)
        char_image = cv2.warpPerspective(img, mat, (IMG_SIZE, IMG_SIZE))

        cv2.imwrite(
            f"./dataset/{char_font_val}/image_{index:03d}_{idx:03d}_char_{char_txt_val:03d}_font_{char_font_val}.png",
            char_image)

    # words = np.swapaxes(np.array(wordBB, np.int32), 0, 2)

    # print("~~~~~~~~~~~~~~~")
    # print(words)
    # print("~~~~~~~~~~~~~~~")
    # for word in words:
    #     cv2.polylines(img, np.asarray([word]), True, color=(0, 255, 0))
