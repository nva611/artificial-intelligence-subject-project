from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2 as cv
import numpy as np
model = tf.keras.models.load_model("symbol.h5")


def symbol(ind):
    symbols = ['!', '+', '0', ')', '(', ',', '-']
    symb = symbols[ind.argmax()]
    return symb


def prediction(image_path):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    plt.imshow(img, cmap='gray')
    img = cv.resize(img, (45, 45))
    norm_image = cv.normalize(
        img, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    norm_image = norm_image.reshape(
        (norm_image.shape[0], norm_image.shape[1], 1))
    case = np.asarray([norm_image])
    pred = model.predict([case])
    return 'Prediction: ' + symbol(pred)


print(prediction("-.jpg"))
