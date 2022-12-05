import cv2


def hello():
    print("nva hello")


def gen_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            detector = cv2.CascadeClassifier(
                'Haarcascades/haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier(
                'Haarcascades/haarcascade_eye.xml')
            faces = detector.detectMultiScale(frame, 1.1, 7)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Draw the rectangle around each face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey),
                                  (ex+ew, ey+eh), (0, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def setData():
    camera = cv2.VideoCapture(0)
    count = 0
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            detector = cv2.CascadeClassifier(
                'Haarcascades/haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier(
                'Haarcascades/haarcascade_eye.xml')
            faces = detector.detectMultiScale(frame, 1.1, 7)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Draw the rectangle around each face
            for (x, y, w, h) in faces:
                count += 1
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey),
                                  (ex+ew, ey+eh), (0, 255, 0), 2)
                cv2.imwrite("FaceRecognize/dataSet/" + "name" + "." + "9" +
                            '.' + str(count) + ".jpg", gray[y:y+h, x:x+w])

            k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
            if k == 27:
                break
            elif count >= 10:
                break

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
