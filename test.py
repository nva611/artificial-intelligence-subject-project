

import os
error = []
for i in range(1, 4):
    print("xoa FaceRecognize/aaaa/An Nguyen Van 1.4."+str(i)+".jpg")
    try:
        os.remove("FaceRecognize/aaaa/An Nguyen Van 1.4."+str(i)+".jpg")
        continue

    except WindowsError:
        print("HUHU")
        continue
