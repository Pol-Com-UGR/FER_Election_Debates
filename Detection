# This file will create a csv file with the located faces in the video files in "/home/javier/Documentos/prueba_debates"

import face_recognition
import pickle
import cv2
import numpy as np
from fer import FER
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt

cascPathface = "haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPathface)
data = pickle.loads(open("face_enc_debates", "rb").read())
videos_folder = "/home/javier/Documentos/prueba_debates"
output_csv = "face_detections.csv"
frame_rate = 30 
emotion_detector = FER(mtcnn=True)
results = []
video_files = glob.glob(os.path.join(videos_folder, "*.mp4"))
for video_file in video_files:
    video_capture = cv2.VideoCapture(video_file)

    contador = 0
    d_contador = {}
    d2_contador = {}

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break  # Break the loop if no frame is captured

        frame_number = video_capture.get(cv2.CAP_PROP_POS_FRAMES)
        contador += 1
        if contador != 25:
            continue
        else:
            contador = 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb)
        names = []
        emotions = []

        for encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Desconocido"

            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                    if name in d_contador:
                        d_contador[name] += 1
                    else:
                        d_contador[name] = 1

                name = max(counts, key=counts.get)

            names.append(name)

        for (x, y, w, h), name in zip(faces, names):
            face_img = frame[y:y+h, x:x+w]
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            emotion_predictions = emotion_detector.detect_emotions(face_img)

            if len(emotion_predictions) > 0:
                emotion_probabilities = emotion_predictions[0]['emotions']
                emotion_label = max(emotion_probabilities, key=emotion_probabilities.get)
            else:
                emotion_label = "Unknown"

            emotions.append(emotion_label)

            # Calculate the timestamp in seconds
            timestamp = frame_number / frame_rate

            # Append the face detection details to the results list
            result = {
                'VideoFile': video_file,
                'Name': name,
                'Timestamp': timestamp,
                'Emotion': emotion_label
            }
            results.append(result)

    video_capture.release()
