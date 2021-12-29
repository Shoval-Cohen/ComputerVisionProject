from keras.preprocessing import image
import cv2
import matplotlib.pyplot as plt

img = image.load_img('15f8U.png', grayscale=True, target_size=(224, 224))
img = image.img_to_array(img, dtype='uint8')