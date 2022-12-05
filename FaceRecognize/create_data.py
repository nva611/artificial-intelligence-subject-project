import cv2
import os
import csv
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

header = ["ID", "Name"]


def writeToCSV(id, name):
    f = open("people.csv", "a", newline='')
    writer = csv.writer(f)
    tup = [id, name]
    writer.writerow(tup)
    f.close()


def idExisted(id):
    url = open("people.csv", "r")
    read_file = csv.reader(url)
    for row in read_file:
        if (row[0] == str(id)):
            return True
    return False


# For each person, enter one numeric face id
face_id = input('\n YOUR ID:  ')
while (True):
    if idExisted(face_id) == False:
        break
    face_id = input('\n ID EXISTED, ENTER YOUR ID:  ')
name = input('\n YOUR NAME:  ')

writeToCSV(face_id, name)

print("\n Waiting...")
# Initialize individual sampling face count
count = 0

while (True):

    ret, img = cam.read()
    img = cv2.flip(img, 1)  # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/" + name + "." + str(face_id) +
                    '.' + str(count) + ".jpg", gray[y:y+h, x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 200:  # Take 200 face sample and stop video
        break

# Do a bit of cleanup
print("\n Successfully created data")
cam.release()
cv2.destroyAllWindows()
