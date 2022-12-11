import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import tensorflow as tf
import pathlib
import PIL
import time
import zipfile
import random
from tensorflow.keras.layers import *
import warnings


TEST_PATH = "test_dataset/"
MAIN_PATH = "train_dataset/train"


HEIGHT, WIDTH = 32, 32
BATCH_SIZE = 32
SPLIT = 0.2
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    validation_split=SPLIT)

train_ds = train_datagen.flow_from_directory(
    MAIN_PATH,
    target_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE,
    subset="training",
    class_mode="categorical",
    shuffle=True
)
class_names = list(train_ds.class_indices.keys())
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255)
test_ds = test_datagen.flow_from_directory(
    "../static/uploads/trafic",
    target_size=(HEIGHT, WIDTH),
    shuffle=False
)


label_df = pd.read_csv("test_labels.csv")
labels = np.array(label_df.label)

model = tf.keras.models.load_model("ok.h5")


def my_predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence


for images, labels in test_ds:
    # print(labels)
    predicted_class, confidence = my_predict(model, images[0])
    # actual_class = class_names[int(labels[i])]
    plt.imshow(images[0])
    plt.title(
        f"Actual: Predicted: {predicted_class}.\n Confidence: {confidence}%")

    plt.axis("off")
    plt.show()
    break
