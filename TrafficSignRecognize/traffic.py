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
MAIN_PATH = "TrafficSignRecognize/train_dataset/train"


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
    "static/uploads/traffic",
    target_size=(HEIGHT, WIDTH),
    shuffle=False
)


label_df = pd.read_csv("TrafficSignRecognize/test_labels.csv")
labels = np.array(label_df.label)

model = tf.keras.models.load_model("TrafficSignRecognize/ok.h5")


def my_predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence


def recognize():
    predicted_class = ""
    confidence = ""
    for images, labels in test_ds:
        # print(labels)
        predicted_class, confidence = my_predict(model, images[0])
        # actual_class = class_names[int(labels[i])]
        break
    if(predicted_class == "GuideSign"):
        return "Guide Sign"
    elif(predicted_class == "M1"):
        return "Command Sign"
    elif(predicted_class =="M4"):
        return "Car Sign"
    elif(predicted_class == "M5"):
        return "Bike Sign"
    elif(predicted_class =="M7"):
        return "Pedestrian Crossing Sign"
    elif(predicted_class == "P1"):
        return "Prohibition Sign"
    elif(predicted_class =="P10_50"):
        return "Speeding Limit Sign"
    elif(predicted_class =="W1"):
        return "Warning Sign"
    return "Unrecognizable"
