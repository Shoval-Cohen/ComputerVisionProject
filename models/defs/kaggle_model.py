import os

import tensorflow as tf
from utils.consts import IMG_SIZE

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print("TensorFlow version:", tf.__version__)

num_classes = 7

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 28x28 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    #     tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    #     tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fourth convolution
    #     tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    #     tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# dot_img_file = '../images/kaggle_font_recog.png'
# tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)


def getModel():
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


getModel()
