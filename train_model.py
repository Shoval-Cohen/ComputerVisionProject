from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard

from utils.consts import IMG_SIZE, num_classes, batch_size
from utils.data_manipulations import preprocess_h5_dataset

file_path = "./resources/SynthText.h5"

images, chars, fonts, _, _ = preprocess_h5_dataset(file_path)

# Scaling
images = np.array(images) / 255.0

model = tf.keras.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32, name='image', ),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    # Flatten the results to feed into a DNN
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    # Classification layer
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

dot_img_file = r'models\images\model.png'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

print(model.summary())

start_time = datetime.now()
print(f"Starting training at {start_time}")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`")

log_folder = "logs/Basic/FinalModel/" + datetime.now().strftime("%Y%m%d-%H%M%S")
callbacks = [TensorBoard(log_dir=log_folder,
                         histogram_freq=1,
                         write_graph=True,
                         write_images=True,
                         update_freq='epoch',
                         profile_batch=2,
                         embeddings_freq=1)]

model.fit(np.array(images),
          np.array(fonts),
          batch_size=batch_size,
          epochs=25,
          callbacks=callbacks)

print(f"Finished training at {(datetime.now() - start_time).total_seconds()}")

model.save("saved_model_with_validation.h5")
