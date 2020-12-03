import cv2
import numpy as np
import face_recognition
import os
from _datetime import datetime

# Read all known image
path = "Resources"
images = []
className = []
myList = os.listdir(path)

# Delete all attendance list
f = open('Attendance.csv', 'r+')
f.truncate(0)
f.write(f'Name,Time')
f.close()

# Add each image to list image
for img in myList:
    curImg = cv2.imread(f'{path}/{img}')
    images.append(curImg)
    className.append(os.path.splitext(img)[0])

# Encoding func to encode all known image
def encodeImages(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeImg = face_recognition.face_encodings(img)[0]
        encodeList.append(encodeImg)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

listEncodeKnown = encodeImages(images)
print("Encode success!")

# Compare image get from camera
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    # Resize image to small to smooth perfomance
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    # Convert to RGB
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    # Get all location of faces in frame
    facesCurFrame = face_recognition.face_locations(imgS)
    # Encode all faces in frame
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(listEncodeKnown, encodeFace)
        facesDis = face_recognition.face_distance(listEncodeKnown, encodeFace)
        # Get min distance
        matchIndex = np.argmin(facesDis)
        # Get index of image if match
        if matches[matchIndex]:
            name = className[matchIndex].upper()
            # Draw rectangle for people
            y1, x2, y2, x1 = faceLoc
            y1, x1, y2, x2 = y1*4, x1*4, y2*4, x2*4
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img, name, (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
            # Mark attendance
            markAttendance(name)
    # Show camera
    cv2.imshow("Webcam", img)
    cv2.waitKey(1)