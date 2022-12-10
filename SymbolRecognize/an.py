from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2 as cv
import numpy as np
train_path = "extracted_images"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    validation_split=0.25
)
train_set = train_datagen.flow_from_directory(
    train_path,
    target_size=(45, 45),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical',
    classes=['!', '+', '0', ')', '(', ',', '-'],
    shuffle=True,
    subset='training',
    seed=123
)

test_set = train_datagen.flow_from_directory(
    train_path,
    target_size=(45, 45),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical',
    classes=['!', '+', '0', ')', '(', ',', '-'],
    shuffle=True,
    subset='validation',
    seed=123
)


def symbol(ind):
    symbols = ['!', '+', '0', ')', '(', ',', '-']
    symb = symbols[ind.argmax()]
    return symb


imgs, labels = next(train_set)


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(cv.cvtColor(img, cv.COLOR_RGB2BGR))
        ax.axis('off')
    plt.tight_layout()
    plt.show()


plotImages(imgs)

model = tf.keras.models.Sequential()

# First Convolutional Block
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5),
          padding='same', activation='relu', input_shape=(45, 45, 1)))
model.add(tf.keras.layers.MaxPool2D(strides=2))

# Second Convolutional Block
model.add(tf.keras.layers.Conv2D(filters=48, kernel_size=(
    5, 5), padding='valid', activation='relu'))
model.add(tf.keras.layers.MaxPool2D(strides=2))

# Classifier Head
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(84, activation='relu'))
model.add(tf.keras.layers.Dense(7, activation='softmax'))

adam = tf.keras.optimizers.Adam(lr=5e-4)
model.compile(optimizer=adam, loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_set, validation_data=test_set, epochs=7)
model.save("symbol.h5")
