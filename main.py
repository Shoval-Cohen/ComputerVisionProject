import cv2
import h5py
import numpy as np

file_path = "./resources/SynthText.h5"
db = h5py.File(file_path, 'r')

im_names = list(db["data"].keys())
for index in np.random.choice(973, 5):
    # for index in [200]:
    im = im_names[index]

    img = db['data'][im][:]

    font = db['data'][im].attrs['font']
    txt = db['data'][im].attrs['txt']
    charBB = db['data'][im].attrs['charBB']
    wordBB = db['data'][im].attrs['wordBB']

    pts = np.swapaxes(np.array(charBB, np.int32), 0, 2)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(pts)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    for char in pts:
        orig = cv2.polylines(img.copy(), np.asarray([char]), True, color=(0, 0, 255))
        cv2.imshow("original", orig)

        ## (1) Crop the bounding rect
        rect = cv2.boundingRect(char)
        x, y, w, h = rect
        cropped = img[y:y + h, x:x + w].copy()

        ## (2) make mask

        char = char - char.min(axis=0)

        mask = np.zeros(cropped.shape[:2], np.uint8)
        cv2.drawContours(mask, [char], -1, (255, 255, 255), -1, cv2.LINE_AA)

        ## (3) do bit-op
        dst = cv2.bitwise_and(cropped, cropped, mask=mask)
        cv2.imshow("dst", dst)

        ## (4) add the white background
        bg = np.ones_like(cropped, np.uint8) * 255

        cv2.bitwise_not(bg, bg, mask=mask)
        dst2 = bg + dst
        cv2.imshow("dst2", dst2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    blank = np.zeros(img.shape)

    words = np.swapaxes(np.array(wordBB, np.int32), 0, 2)

    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print(words)
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # for word in words:
    #     cv2.polylines(img, np.asarray([word]), True, color=(0, 255, 0))

    #
    cv2.imwrite(f"./imgs/{index}.png", img)
    # print("font", font)
    # print("txt", txt)
    # print("charBB", charBB)
    # print("wordBB", wordBB)
    cv2.waitKey(0)
