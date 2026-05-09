# this doesn't average.... just finds the feature values in each frame
import cv2
import numpy as np
import dlib
import os

video_path = r"/mnt/xdrive/sambit/project/data/AP1/101_AP1.mp4"
output_dir = r"/mnt/xdrive/sambit/project/data/video/AP1_vdo_feat_demo"
os.makedirs(output_dir, exist_ok=True)

predictor_path = r"/mnt/xdrive/sambit/project/codes/video_features_extraction/models/shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

cap = cv2.VideoCapture(video_path)
features = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) > 0:
        landmarks = predictor(gray, faces[0])
        vec = [timestamp]
        for i in range(68):
            vec.append(landmarks.part(i).x)
            vec.append(landmarks.part(i).y)
    else:
        vec = [timestamp] + [0] * 136

    features.append(vec)

cap.release()

features = np.array(features)

# Create column names
columns = ["time_sec"]
for i in range(68):
    columns.append(f"x{i}")
    columns.append(f"y{i}")

# Print column names
print("Columns:")
print(columns)

out_csv = os.path.join(output_dir, "101_AP1_visual.csv")
header = ",".join(columns)

np.savetxt(out_csv, features, delimiter=",", header=header, comments="")

print(f"Saved: {out_csv}")
print("Shape:", features.shape)
