import os
import pandas as pd
from PIL import Image

import numpy as np



input_path = ".\\PatientData"

folders = []
csv_files = []

for i in os.listdir(input_path):
    if "Patient" in i:
        if "Labels" not in i:
            folders.append(i)
        else:
            csv_files.append(i)

# patient_names = folders.copy()

for i in range(len(folders)):
    folders[i] = os.path.join(input_path, folders[i])

for i in range(len(csv_files)):
    csv_files[i] = os.path.join(input_path, csv_files[i])

folders = sorted(folders)
csv_files = sorted(csv_files)



labels_list = []
image_names = []

for i in range(len(folders)):

    files_list = os.listdir(folders[i])

    for file_name in files_list:
        if "thresh" in file_name:
            image_names.append(os.path.join(folders[i], file_name))

    csv_path = csv_files[i]

    df = pd.read_csv(csv_path)

    for r, v in df.iterrows():
        if v[1] == 0:
            pass
        else:
            df.iloc[r][1] = 1
    
    labels_list += list(df['Label'])


images_list = []

image_width = 224
image_height = 224

for i in range(len(image_names)):
    img = Image.open(image_names[i])

    img = img.resize((image_width, image_height))

    images_list.append(np.asarray(img))



test_path = ".\\testPatient"

test_images_path = os.path.join(test_path, "test_Data")

test_csv_path = os.path.join(test_path, "test_Labels.csv")

test_file_list = os.listdir(test_images_path)

test_images = []

for file_name in test_file_list:
    if "thresh" in file_name:
        test_images.append(os.path.join(test_images_path, file_name))

test_df = pd.read_csv(test_csv_path)

for r, v in test_df.iterrows():
    if v[1] == 0:
        pass
    else:
        test_df.iloc[r][1] = 1

test_labels_list = list(test_df['Label'])

test_images_list = []

for i in range(len(test_images)):
    img = Image.open(test_images[i])
    img = img.resize((image_width, image_height))
    test_images_list.append(np.asarray(img))
    


import tensorflow as tf

X_train = np.array(images_list.copy()) / 255
y_train = np.array(labels_list.copy())

X_train.reshape(-1, image_height, image_width, 1)

X_test = np.array(test_images_list.copy()) / 255
y_test = np.array(test_labels_list.copy())

X_test.reshape(-1, image_height, image_width, 1)



model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(64, (3,3), input_shape=(image_height, image_width, 3), padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Conv2D(512, (3,3), padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(512, (3,3), padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(512, (3,3), padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(512, (3,3), padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Conv2D(512, (3,3), padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(512, (3,3), padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(512, (3,3), padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(512, (3,3), padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(2))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
model.compile(optimizer = optimizer , loss = "binary_crossentropy" , metrics = ['accuracy'])

# print(model.summary())

model.fit(X_train,y_train, epochs = 10, validation_data = (X_test, y_test))

model.save("model_Assignment_3")
