import cv2
import numpy as np
import face_recognition

# Read image
imgElon = face_recognition.load_image_file("Resources/Elon Musk.jpg")
imgElonTest = face_recognition.load_image_file("Resources/Bill Gates.jpeg")

# Convert to RGB
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgElonTest = cv2.cvtColor(imgElonTest, cv2.COLOR_BGR2RGB)

# Get the location of face detected
faceLocation = face_recognition.face_locations(imgElon)[0]
faceTestLocation = face_recognition.face_locations(imgElonTest)[0]

# Encode image
encodeElon = face_recognition.face_encodings(imgElon)[0]
encodeTest = face_recognition.face_encodings(imgElonTest)[0]

# Check for result and compare distance between encode
result = face_recognition.compare_faces([encodeElon], encodeTest)
faceDis = face_recognition.face_distance([encodeElon], encodeTest)

print(result, faceDis)

# Draw rect for face and label for result
cv2.rectangle(imgElon, (faceLocation[3], faceLocation[0]), (faceLocation[1], faceLocation[2]), (255, 0, 255), 2)
cv2.rectangle(imgElonTest, (faceTestLocation[3], faceTestLocation[0]), (faceTestLocation[1], faceTestLocation[2]), (255, 0, 255), 2)
cv2.putText(imgElonTest, f'{result} {faceDis}',(50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 50, 255), 2)

# Show image
cv2.imshow("Elon", imgElon)
cv2.imshow("Elon test", imgElonTest)

cv2.waitKey(0)
cv2.destroyAllWindows()