import os

import tensorflow as tf
from tensorflow.keras import Sequential, layers

from utils.consts import IMG_SIZE

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print("TensorFlow version:", tf.__version__)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



model = Sequential()

# Cu Layers 
model.add(layers.Conv2D(64, 24, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(128, 12, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#
# model.add(layers.
#           Conv2DTranspose(128, (24, 24), strides=(2, 2), activation='relu', padding='same',
#                           kernel_initializer='uniform'))
# model.add(layers.UpSampling2D(size=(2, 2)))
# #
# model.add(layers.
#           Conv2DTranspose(64, (12, 12), strides=(2, 2), activation='relu', padding='same',
#                           kernel_initializer='uniform'))
# model.add(layers.UpSampling2D(size=(2, 2)))
#
# # Cs Layers
model.add(layers.Conv2D(256, kernel_size=(12, 12), activation='relu'))

# model.add(layers.Conv2D(256, kernel_size=(12, 12), activation='relu'))
#
# model.add(layers.Conv2D(256, kernel_size=(12, 12), activation='relu'))

model.add(layers.Flatten())

model.add(layers.Dense(2048, activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(2048, activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(2383, activation='relu'))


print("Building")
model.build(input_shape=(None, IMG_SIZE, IMG_SIZE, 3))
print("Finished")
dot_img_file = r'C:\devl\study\ComputerVisionProject\models\images\another.png'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)


def getModel():
    return model
