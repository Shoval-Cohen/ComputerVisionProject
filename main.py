import cv2
import h5py
import imutils
import numpy as np

file_path = "./resources/SynthText.h5"
db = h5py.File(file_path, 'r')

im_names = list(db["data"].keys())
image_num = 1
for im in im_names:

    print(f"working on image {image_num} with {im} name")
    # for index in np.random.choice(973, 5):
    # for index in [200]:
    # im = im_names[index]

    img = db['data'][im][:]

    font = db['data'][im].attrs['font']
    txt = db['data'][im].attrs['txt']
    charBB = db['data'][im].attrs['charBB']
    wordBB = db['data'][im].attrs['wordBB']

    pts = np.swapaxes(charBB, 0, 2)
    #
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print(pts)
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    for char in pts:
        c = 0
        original_char = char.copy()
        char = np.asarray(char, np.int32)
        orig = cv2.polylines(img.copy(), np.asarray([char]), True, color=(0, 0, 255))
        # cv2.imshow("original", orig)

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
        c += 1
        if char_image is not None:
            # Check about use contour
            # cv2.imshow("a", char_image)
            char_image = cv2.GaussianBlur(char_image, (3, 3), 1)
            char_image = cv2.Canny(char_image, 100, 200)
            char_image = cv2.resize(char_image, (char_image.shape[1] * 4, char_image.shape[0] * 4))
            # Add the char letter here
            cv2.imwrite(f"./imgs/image_{image_num}_{c}.png", char_image)


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

    image_num += 1

    # words = np.swapaxes(np.array(wordBB, np.int32), 0, 2)

    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print(words)
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # for word in words:
    #     cv2.polylines(img, np.asarray([word]), True, color=(0, 255, 0))
