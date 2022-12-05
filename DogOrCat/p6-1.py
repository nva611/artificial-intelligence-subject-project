import cv2
import tensorflow as tf

# will use this to convert prediction num to string value
CATEGORIES = ["Dog", "Cat"]


def prepare(filepath):
    IMG_SIZE = 50  # 50 in txt-based
    # read in the image, convert to grayscale
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    # resize image to match model's expected sizing
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    # return the image with shaping that TF wants.
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("64x3-CNN.model")


prediction = model.predict([prepare('meo.jpg')])
print(prediction)  # will be a list in a list.
print(CATEGORIES[int(prediction[0][0])])
