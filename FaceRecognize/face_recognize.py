import cv2
import os
import numpy as np
from PIL import Image
import csv


header = ["ID", "Name"]


def test(name):
    print("prepare", name)


def deleteAll():
    url = open("FaceRecognize/people.csv", "r")
    read_file = csv.reader(url)
    for row in read_file:
        for i in range(1, 101):
            try:
                os.remove("FaceRecognize/dataSet/" +
                          row[1] + "." + row[0] + "." + str(i) + ".jpg")
            except WindowsError:
                continue
    with open('FaceRecognize/people.csv', 'rb') as inp, open('FaceRecognize/people.csv', 'wb') as out:
        writer = csv.writer(out)
        for row in csv.reader(inp):
            writer.writerow(row)

    inp.close()
    out.close()


def getList():
    id = []
    name = []
    url = open("FaceRecognize/people.csv", "r")
    profile = None
    read_file = csv.reader(url)
    for row in read_file:
        id.append(row[0])
        name.append(row[1])
    print("FACE: ", id, name)
    url.close()
    return id, name


def getName(id):
    url = open("FaceRecognize/people.csv", "r")
    profile = None
    read_file = csv.reader(url)
    for row in read_file:
        if (row[0] == str(id)):
            profile = row[1]

    url.close()
    return profile


def getImagesAndLabels(path):
    detector = cv2.CascadeClassifier(
        "Haarcascades/haarcascade_frontalface_default.xml")
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
        img_numpy = np.array(PIL_img, 'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)

    return faceSamples, ids


def writeToCSV(id, name):
    if idExisted(id):
        return
    f = open("FaceRecognize/people.csv", "a", newline='')
    writer = csv.writer(f)
    tup = [id, name]
    writer.writerow(tup)
    f.close()


def idExisted(id):
    url = open("FaceRecognize/people.csv", "r")
    read_file = csv.reader(url)
    for row in read_file:
        if (row[0] == str(id)):
            return True
    return False


def createData(face_id, name):
    print("+++++++++++++++++++++", name)
    count = 0
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height

    face_detector = cv2.CascadeClassifier(
        'Haarcascades/haarcascade_frontalface_default.xml')
    writeToCSV(face_id, name)
    while (True):
        ret, img = cam.read()
        img = cv2.flip(img, 1)  # flip video image vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:

            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            count += 1

            # Save the captured image into the datasets folder
            cv2.imwrite("FaceRecognize/dataSet/" + name + "." + str(face_id) +
                        '.' + str(count) + ".jpg", gray[y:y+h, x:x+w])

            #cv2.imshow('image', img)
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 100:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    print("\n Successfully created data")
    cam.release()
    cv2.destroyAllWindows()
    # lay anh
    print("+++++++++++++++++++++ ENDDDDDDDDDD", name)


def trainModel():
    path = 'FaceRecognize/dataSet'

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces, ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))
    if not os.path.exists('FaceRecognize/recognizer'):
        os.makedirs(recognizer)
    # Save the model into trainer/trainer.yml
    # recognizer.save() worked on Mac, but not on Pi
    recognizer.write('FaceRecognize/recognizer/trainer.yml')
    print("\n [INFO] {0} faces trained. Exiting Program".format(
        len(np.unique(ids))))
    cv2.destroyAllWindows()
    return True


def recognize():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('FaceRecognize/recognizer/trainer.yml')
    cascadePath = "FaceRecognize/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)

    font = cv2.FONT_HERSHEY_SIMPLEX

    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video widht
    cam.set(4, 480)  # set video height

    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)

    while True:

        ret, live_frame = cam.read()
        # img = cv2.flip(img, -1) # Flip vertically

        gray = cv2.cvtColor(live_frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:

            cv2.rectangle(live_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            name = ""
            # Check if confidence is less them 100 ==> "0" is perfect match
            if (confidence < 100):
                name = getName(id)
                confidence = "  {0}%".format(round(100 - confidence))

            else:
                name = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(live_frame, str(name), (x+5, y-5),
                        font, 1, (255, 255, 255), 2)
            cv2.putText(live_frame, str(confidence), (x+5, y+h-5),
                        font, 1, (255, 255, 0), 1)

        # cv2.imshow('camera', img)
        ret, buffer = cv2.imencode('.jpg', live_frame)
        live_frame = buffer.tobytes()
        # k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        # if k == 27:
        #     break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + live_frame + b'\r\n')

    # Do a bit of cleanup
    # print("\n [INFO] Exiting Program and cleanup stuff")
    # cam.release()
    # cv2.destroyAllWindows()


def createDataSet(face_id, name):
    count = 0
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height

    face_detector = cv2.CascadeClassifier(
        'Haarcascades/haarcascade_frontalface_default.xml')
    writeToCSV(face_id, name)
    while (True):
        ret, img = cam.read()
        img = cv2.flip(img, 1)  # flip video image vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:

            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            count += 1

            # Save the captured image into the datasets folder
            cv2.imwrite("FaceRecognize/dataSet/" + name + "." + str(face_id) +
                        '.' + str(count) + ".jpg", gray[y:y+h, x:x+w])

        if count >= 99:
            break
    print("\n Successfully created data")
    cam.release()
    cv2.destroyAllWindows()
