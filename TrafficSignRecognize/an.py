import os
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
warnings.filterwarnings('ignore')
MAIN_PATH = "train_dataset/train"
TEST_PATH = "test_dataset/test/"
CLASSES = os.listdir(MAIN_PATH)
NUM_CLASSES = len(CLASSES)

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

val_ds = train_datagen.flow_from_directory(
    MAIN_PATH,
    target_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE,
    subset="validation",
    class_mode="categorical",
    shuffle=True
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255)
test_ds = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=(HEIGHT, WIDTH),
    shuffle=False
)


def create_model():
    vgg16 = tf.keras.applications.VGG16(
        include_top=False, weights='imagenet', input_shape=[HEIGHT, WIDTH, 3])

    x = vgg16.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.GaussianDropout(0.4)(x)
    outputs = tf.keras.layers.Dense(
        NUM_CLASSES, activation="softmax", dtype='float32')(x)

    model = tf.keras.Model(vgg16.input, outputs)
    return model


model = create_model()


def compile_model(model, lr=0.0001):

    optimizer = tf.keras.optimizers.Adam(lr=1e-4)

    loss = tf.keras.losses.CategoricalCrossentropy()

    metrics = [
        tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy')
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def create_callbacks():

    cpk_path = './best_model.h5'

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=cpk_path,
        monitor='val_categorical_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1,
    )

    reducelr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_categorical_accuracy',
        mode='max',
        factor=0.1,
        patience=3,
        verbose=0
    )

    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor='val_categorical_accuracy',
        mode='max',
        patience=10,
        verbose=1
    )

    callbacks = [checkpoint, reducelr, earlystop]

    return callbacks


EPOCHS = 60
VERBOSE = 1

tf.keras.backend.clear_session()

with tf.device('/device:GPU:0'):

    model = create_model()
    model = compile_model(model, lr=0.0001)

    callbacks = create_callbacks()

    history = model.fit(train_ds,
                        epochs=EPOCHS,
                        callbacks=callbacks,
                        validation_data=val_ds,
                        verbose=VERBOSE)
    model.save('an.h5')
