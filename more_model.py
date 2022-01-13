import os

import tensorflow as tf
from tensorflow.keras import Sequential, layers

from utils.consts import IMG_SIZE

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

model = Sequential()
model.add(layers.Convolution2D(32, kernel_size=(3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(layers.Activation('relu'))
model.add(layers.Convolution2D(64, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Convolution2D(64, (3, 3), padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Convolution2D(64, 3, 3))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(512))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))

print("Building")
# model.build(input_shape=(None, IMG_SIZE, IMG_SIZE, 3)

# let's train the model using SGD + momentum (how original).
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer=sgd,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'],
              run_eagerly=True)

print("Finished")
dot_img_file = r'C:\devl\study\ComputerVisionProject\models\images\more.png'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)


def getModel():
    return model
