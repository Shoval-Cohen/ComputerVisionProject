import csv
import os
from datetime import datetime

import numpy as np
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow import keras

from utils.consts import num_classes, font_dict
from utils.data_manipulations import preprocess_h5_dataset

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

file_path = "../resources/SynthText_test.h5"

is_training = False

chars_images, chars, fonts, words, txt_img_names = preprocess_h5_dataset(file_path, is_training=is_training)

model = keras.models.load_model('../resources/saved_multi_input_model.h5')

chars_amount = len(chars)

start_time = datetime.now()
print(f"Starting prediction at {start_time}")

predicted_fonts_proba = model.predict([np.array(chars_images), np.array(chars)])

print(f"Finished prediction at {datetime.now()}",
      f"Took {(datetime.now() - start_time).total_seconds()}s")

selected_fonts = np.zeros(chars_amount)

idx = 0
for word in words:
    word_font_votes = np.zeros(num_classes)
    for char_idx in range(idx, idx + len(word)):
        best_font_for_char = np.argmax(predicted_fonts_proba[char_idx])
        word_font_votes[best_font_for_char] += 1
    selected_fonts[idx:idx + len(word)] = np.argmax(word_font_votes)
    idx += len(word)

result_file = "results_multi_input.csv"
start_time = datetime.now()

print(f"Starting to write the results to {result_file} at {start_time}")
with open(result_file, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow([" ", "image", "char", "b'Raleway", "b'Open Sans", "b'Roboto", "b'Ubuntu Mono", "b'Michroma",
                     "b'Alex Brush", "b'Russo One"])
    for row_index in range(chars_amount):
        row = [row_index, txt_img_names[row_index], chr(chars[row_index])]
        row.extend(np.int32(to_categorical(selected_fonts[row_index], num_classes=num_classes)))
        writer.writerow(row)
print(f"Finished to write the results to {result_file} at {datetime.now()}",
      f"Took {(datetime.now() - start_time).total_seconds()}s")

if is_training:
    print("Accuracy:")
    print(accuracy_score(fonts, selected_fonts))
    print("Confusion Matrix:")
    print(confusion_matrix(fonts, selected_fonts))
    ConfusionMatrixDisplay.from_predictions(fonts,
                                            selected_fonts,
                                            display_labels=[fnt.decode("utf8") for fnt in font_dict.keys()])
    acc_percents = accuracy_score(fonts, selected_fonts) * 100
    plt.title(f"Accuracy: {acc_percents:.2f}%")
    plt.xticks(rotation=90)
    plt.show()
