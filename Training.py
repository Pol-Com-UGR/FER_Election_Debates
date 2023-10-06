# This Python code will train face_recognition to recognise the faces in the folders within "Imagenes_Cand_Lim"
# The result is the file "face_enc_debates_lim" which will be used by opencv to recognise faces in video files
from imutils import paths
import face_recognition
import pickle
import cv2
import os

imagePaths = list(paths.list_images('Imagenes_Cand_Lim'))
knownEncodings = []
knownNames = []
for (i, imagePath) in enumerate(imagePaths):
    name = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb,model='hog')
    encodings = face_recognition.face_encodings(rgb, boxes)
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)
data = {"encodings": knownEncodings, "names": knownNames}
f = open("face_enc_debates_lim", "wb")
f.write(pickle.dumps(data))
f.close()
