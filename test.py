import csv

import numpy as np
from keras.utils.np_utils import to_categorical
from tensorflow import keras

from utils.consts import num_classes
from utils.data_manipulators import preprocess_h5_dataset

file_path = "resources/SynthText.h5"

chars_images, chars, fonts, words, big_img_names = preprocess_h5_dataset(file_path)

print("Predicting")
model = keras.models.load_model('saved_model_0.h5')

chars_amount = len(chars)
predicted_fonts_proba = model.predict(np.array(chars_images))

selected_fonts = np.zeros(chars_amount)

idx = 0
for word in words:
    word_font_votes = np.zeros(num_classes)
    for char_idx in range(idx, idx + len(word)):
        best_font_for_char = np.argmax(predicted_fonts_proba[char_idx])
        word_font_votes[best_font_for_char] += 1
    selected_fonts[idx:idx + len(word)] = np.argmax(word_font_votes)
    idx += len(word)

with open(f'my_results.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow([" ", "image", "char", "b'Raleway", "b'Open Sans", "b'Roboto", "b'Ubuntu Mono", "b'Michroma",
                     "b'Alex Brush", "b'Russo One"])
    for row_index in range(chars_amount):
        row = [row_index, big_img_names[row_index], chr(chars[row_index])]
        row.extend(np.int32(to_categorical(selected_fonts[row_index], num_classes=num_classes)))
        print(row)
        writer.writerow(row)

