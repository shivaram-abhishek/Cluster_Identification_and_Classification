# %%

import os
import tensorflow as tf
import pandas as pd
from PIL import Image
import numpy as np

# %%

test_path = ".\\testPatient"

image_width = 224
image_height = 224

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

# %%

X_test = np.array(test_images_list.copy()) / 255
y_test = np.array(test_labels_list.copy())

# %%

model = tf.keras.models.load_model("model_Assignment_3")

# %%

result = []

for i in range(len(test_images_list)):

    img = test_images_list[i] / 255.0
    img = img.reshape(1, image_height, image_width, 3)
    label = model.predict(img, verbose=0)
    if label[0][0] <= 0.5:
        result.append([i+1, 0])
    else:
        result.append([i+1, 1])

result_df = pd.DataFrame(result, columns=[["IC", "Label"]], index=None)
result_df.to_csv("Results.csv", index=None)

# %%

test_results = []

for i in range(len(y_test)):
    test_results.append([i+1, y_test[i]])

test_results_df = pd.DataFrame(test_results, columns=[["IC", "Label"]])

tp = 0
tn = 0
fp = 0
fn = 0

for i in range(len(result)):
    if result[i][1] == 1 and test_results[i][1] == 1:
        tp += 1
    elif result[i][1] == 0 and test_results[i][1] == 0:
        tn += 1
    elif result[i][1] == 1 and test_results[i][1] == 0:
        fp += 1
    elif result[i][1] == 0 and test_results[i][1] == 1:
        fn += 1

calc_accuracy = (tp+tn) / (tp+fn+tn+fp)
calc_specificity = tn / (tn+fp)
calc_sensitivity = tp / (tp+fn)
calc_precision = tp / (tp+fp)
calc_recall = tp / (tp+fn)
calc_f1 = (2*calc_precision*calc_recall) / (calc_precision+calc_recall)

metrics_list = [["Accuracy", calc_accuracy], ["Precision", calc_precision], ["Specificity", calc_specificity], ["Sensitivity", calc_sensitivity], ["Recall", calc_recall], ["F1 Score", calc_f1]]

metrics_df = pd.DataFrame(metrics_list, columns=[["Metric", "Measure"]], index=None)
metrics_df.to_csv("Metrics.csv", index=None)
