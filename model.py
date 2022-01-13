import datetime
import sys

import tensorflow as tf
from tensorflow.keras import layers

import more_model
from utils.consts import IMG_SIZE, num_classes, batch_size
from utils.data_manipulators import split_and_prepare

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

data_dir = "./dataset"

multi_input = (sys.argv[0] == "True")

ds = tf.data.Dataset.list_files("dataset/*/*")

print(f"MultiInput={multi_input}")
print(f"Trying to prepare {datetime.datetime.now()}")
train_ds, val_ds = split_and_prepare(ds, multi_input=multi_input)
print(f"Finished preparation {datetime.datetime.now()}")

# cnn = tf.keras.Sequential([
#     # hub.KerasLayer(model_handle, trainable=True),
#     tf.keras.layers.Dropout(rate=0.2),
#     tf.keras.layers.Dense(num_classes,
#                           kernel_regularizer=tf.keras.regularizers.l2(0.0001))
# ])
used_model = more_model.getModel()
# used_model = another_model.getModel()

if multi_input:
    print("Crating model with Multiple inputs")
    image_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32, name='image')
    x = used_model(image_input)
    # x = layers.Flatten()(x)

    char_input = layers.Input(shape=(1,), dtype=tf.float32, name='char')

    combined = layers.concatenate([x, char_input])
    combined = layers.Dense(64, activation='relu')(combined)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=[image_input, char_input], outputs=outputs)

else:
    print("Crating model with just images as inputs")
    model = tf.keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32, name='image'),
        used_model,
        layers.Dense(num_classes, activation='softmax')
    ])

# # Compile and fit the model
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=0.1,
#     decay_steps=10000,
#     decay_rate=0.9)

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy'])

dot_img_file = f'models/images/model_multi_input_{multi_input}.png'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

print(model.summary())

print("Let's go!")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    batch_size=batch_size
)

model.save("saved_model.h5")
