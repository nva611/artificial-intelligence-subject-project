# from FaceRecognize import create_data, train_model, recognize
from utils import my_camera
from FaceRecognize import face_recognize
import matplotlib.pyplot as plt
import tensorflow as tf
from flask import Flask, render_template, Response, request, flash, redirect, url_for, make_response, jsonify
import cv2
import numpy as np
import urllib.request
import os
from werkzeug.utils import secure_filename
import _thread
import threading
import time
exitFlag = 0


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = "aaa"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

# DOG OR CAT


# will use this to convert prediction num to string value
CATEGORIES = ["Dog", "Cat"]

#############
def symbol(ind):
    symbols = ['!', '+', '0', ')', '(', ',', '-']
    symb = symbols[ind.argmax()]
    return symb


def prediction(image_path):
    model = tf.keras.models.load_model("./SymbolRecognize/symbol.h5")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(img, cmap='gray')
    img = cv2.resize(img, (45, 45))
    norm_image = cv2.normalize(
        img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_image = norm_image.reshape(
        (norm_image.shape[0], norm_image.shape[1], 1))
    case = np.asarray([norm_image])
    pred = model.predict([case])
    return '' + symbol(pred)

# ==============================
def dogcat_recognize(filepath):
    IMG_SIZE = 50  # 50 in txt-based
    # read in the image, convert to grayscale
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    # resize image to match model's expected sizing
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    # return the image with shaping that TF wants.
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


###########################################
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')

# ================================ START DOG OR CAT =================================


@app.route('/dogorcat')
def getDogOrCat():
    return render_template('dogorcat.html')


@app.route('/dogorcat', methods=['POST'])
def recognizeDogOrCat():

    model = tf.keras.models.load_model("./DogOrCat/64x3-CNN.model")
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'dogorcat.jpg'))
        # print('upload_image filename: ' + filename)
        # flash('Image successfully uploaded and displayed below')
        prediction = model.predict(
            [dogcat_recognize('static/uploads/dogorcat.jpg')])
        # print(prediction)  # will be a list in a list.
        result = CATEGORIES[int(prediction[0][0])]  # tra result cho response
        return render_template('dogorcat.html', filename="dogorcat.jpg", result=result)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display')
def display_image():
    return redirect(url_for('static', filename='uploads/' + 'dogorcat.jpg'), code=301)

@app.route('/displaysymbol')
def display_symbol():
    return redirect(url_for('static', filename='uploads/' + 'symbol.jpg'), code=301)
# ================================ END DOG OR CAT ================================


# ================================ START FACE RECOGNIZE ================================


@app.route('/face')
def face():
    id, name = face_recognize.getList()
    print("TRA VE", id, name)
    return render_template('facerecognize.html', id=id, name=name)

@ app.route('/create_data', methods=['POST'])
def create_data():
    id = request.form.get('id')
    name = request.form.get('name')
    result = face_recognize.idExisted(id)
    if (result == True):
        flash('Id already exists please choose another id')
        return redirect("/face")

    face_recognize.writeToCSV(id, name)
    face_recognize.createDataSet(id, name)
    face_recognize.trainModel()
    return redirect(url_for("result_face_recognize", result=1))


@app.route('/result_face_recognize', methods=['POST', 'GET'])
def result_face_recognize():
    return render_template('resultFaceRecognize.html')


@app.route('/face_recognize_live', methods=['POST', 'GET'])
def face_recognize_live():
    return Response(face_recognize.recognize(), mimetype='multipart/x-mixed-replace; boundary=frame')
# face_recognize.recognize()
@app.route('/video_feed')
def video_feed():
    # face_recognize.createData(id, name)
    return Response(my_camera.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# =============================== Symbol===============================
@app.route('/symbol')
def getSymbol():
    return render_template('symbol.html')

@app.route('/symbol', methods=['POST'])
def recognizeSymbol():

    model = tf.keras.models.load_model("./SymbolRecognize/symbol.h5")
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'symbol.jpg'))
        # print('upload_image filename: ' + filename)
        # flash('Image successfully uploaded and displayed below')
        result =  prediction('static/uploads/symbol.jpg')
        return render_template('symbol.html', filename="symbol.jpg", result=result)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)   



# =============================== END FACE RECOGNIZE =============================


@app.route('/function/<FUNCTION>')
def command(FUNCTION=None):

    # exec(FUNCTION.replace("<br>", "\n"))\
    print(FUNCTION)
    result = ""
    try:
        result = eval(FUNCTION.replace("<br>", "\n"))
    except:
        print("ERORR")

    return ""


if __name__ == '__main__':
    app.run(debug=True)


