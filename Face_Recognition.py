# Libraries for project
import os
from datetime import datetime

import cv2
from cv2 import flip
import face_recognition as fr
import numpy as np


path = 'pics'  # picture file path
imgs = []  # numpy array list style for all the pics available
classNames = []  # for names from images
myList = os.listdir(path)  # for grabbing the list of name from image name
print(myList)  # print list

for cl in myList:  # class
    curImg = cv2.imread(f'{path}/{cl}')  # image read
    imgs.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


# face encoding
def findEncodings(imgs):  # encode
    encodeList = []
    for img in imgs:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # detect color
        encode = fr.face_encodings(img)[0]  # training
        encodeList.append(encode)
    return encodeList


# display name under the detection box
def markAttend(name):
    with open('Attend.csv', 'r+') as f:
        mydatalist = f.readlines()
        namelist = []
        for line in mydatalist:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now().time()
            now2 = datetime.now().date()
            Timestr = now.strftime('%I:%M:%S %p')  # Time in 12-hours formate
            dtstr = now2.strftime('%d/%b/%y')  # date
            f.writelines(f'\n{name},\t\t{Timestr},\t{dtstr}')
        # print(MyDataList)

encodeListKnown = findEncodings(imgs)
print('Encode Complete')  # print to encode completion msg

# Cascade Classifier
facecas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# capturing Video with webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
	
# flipVertical = cv2.flip(cap, 0)

# video Resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 864)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Decrease or limit the frame rate/buffer size
cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)

# Reading webcam for realtime recognition
while True:
    ret, img = cap.read()
    img = cv2.flip(img, +1)
    ImgS = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facecas.detectMultiScale(ImgS, scaleFactor=1.05, minNeighbors=5)
    # success, img = cap.read()
    ImgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    # ImgS = cv2.cvtColor(ImgS, cv2.COLOR_BGR2RGB)
    faceCurFrame = fr.face_locations(ImgS)
    encodesCurFrame = fr.face_encodings(ImgS, faceCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, faceCurFrame):
        matches = fr.compare_faces(encodeListKnown, encodeFace)
        faceDis = fr.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:  # matching
            name = classNames[matchIndex].upper()
            # print(name)
            # rectangle box around the face
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttend(name)
        else:
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)

    cv2.imshow('Face Recognition & Attendance System', img)
    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()
