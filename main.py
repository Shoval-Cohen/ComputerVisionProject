import os
os.makedirs("dataset")

for i in range(7):
  os.makedirs(f"dataset/{i}")


import cv2
import h5py
import numpy as np

file_path = "./resources/SynthText.h5"
db = h5py.File(file_path, 'r')

im_names = list(db["data"].keys())
font_dict = {
    b'Roboto': 0,
    b'Michroma':1,
    b'Raleway': 2,
    b'Alex Brush':3,
    b'Ubuntu Mono': 4,
    b'Russo One': 5,
    b'Open Sans':6
}




for index in range(973):
    # if True:
    #     index = 206

    im = im_names[index]
    # print(f"working on image {index} with name {im}")

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

        # print(f"index is {idx}")
        original_char = char.copy()
        char = np.asarray(char, np.int32)
        char = np.where(char > 0, char, 0)
        # orig = cv2.polylines(img.copy(), np.asarray([char]), True, color=(0, 0, 255))
        # cv2_imshow(orig)

        ## (1) Crop the bounding rect
        rect = cv2.boundingRect(char)
        x, y, w, h = rect
        cropped = img[y:y + h, x:x + w].copy()

        ## (2) make mask

        char = char - char.min(axis=0)

        mask = np.zeros(cropped.shape[:2], np.uint8)
        cv2.drawContours(mask, [char], -1, (255, 255, 255), -1, cv2.LINE_AA)

        ## (3) do bit-op
        char_image = cv2.bitwise_and(cropped, cropped, mask=mask)

        # Check about use contour
        # cv2.imshow("a", char_image)
        # char_image = cv2.GaussianBlur(char_image, (3, 3), 1)
        # char_image = cv2.Canny(char_image, 100, 200)
        # char_image = cv2.resize(char_image, (char_image.shape[1] * 4, char_image.shape[0] * 4))

        # Add the char letter here
        # print(f"Image {index} letter num: {idx} with value: {chr(char_txt_val)} and font num: {char_font_val}")
        # cv2_imshow(char_image)
        cv2.imwrite(
            f"./dataset/{char_font_val}/image_{index:03d}_{idx:03d}_char_{char_txt_val:03d}_font_{char_font_val}.png",
            char_image)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # ## Rotate without cutoff
        #
        # # calc angle
        # x0, y0 = original_char[0]
        # x1, y1 = original_char[1]
        # x3, y3 = original_char[-1]
        #
        # cv2.circle(orig, (int(x0), int(y0)), 8, color=(0, 255, 0))
        # cv2.circle(orig, (int(x1), int(y1)), 8, color=(0, 180, 0))
        # cv2.circle(orig, (int(x3), int(y3)), 8, color=(0, 90, 0))
        # cv2.imshow("original", orig)
        #
        # horizontal_angle = np.rad2deg(np.arctan((y1 - y0) / (x1 - x0))) if x1 != x0 else 0
        # vertical_angle = np.rad2deg(np.arctan((x0 - x3) / (y0 - y3))) if y0 != y3 else 0
        # print("horizontal_angle")
        # print(horizontal_angle)
        # print("vertical_angle")
        # print(vertical_angle)
        # angle = max(horizontal_angle, vertical_angle, key=abs)
        # print("angle")
        # print(angle)
        #
        # rotated = imutils.rotate_bound(char_image, angle)
        # cv2.imshow(f"Rotated (Correct {angle})", rotated)

    # words = np.swapaxes(np.array(wordBB, np.int32), 0, 2)

    # print("~~~~~~~~~~~~~~~")
    # print(words)
    # print("~~~~~~~~~~~~~~~")
    # for word in words:
    #     cv2.polylines(img, np.asarray([word]), True, color=(0, 255, 0))
