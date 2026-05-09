# # My test_code for feature_extn on one file.

# import cv2
# import numpy as np
# import dlib
# import os

# video_path = r"C:\Users\sambi\Documents\data\AP1\101_AP1.mp4"
# out_dir = r"C:\Users\sambi\Documents\data\video\AP1_vdo_feat_demo"
# os.makedirs(out_dir, exist_ok=True)

# predictor_path = r"C:\Users\sambi\Documents\Sem8\Project\codes\video_features_extraction\models\shape_predictor_68_face_landmarks.dat"

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(predictor_path)

# cap = cv2.VideoCapture(video_path)

# features = []
# last_sec = -1

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     current_sec = int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)

#     # Procesing only 1FPS
#     if current_sec != last_sec:
#         last_sec = current_sec

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = detector(gray)

#         if len(faces) > 0:
#             landmarks = predictor(gray, faces[0])
#             vec = []
#             for i in range(68):
#                 vec.append(landmarks.part(i).x)
#                 vec.append(landmarks.part(i).y)
#             features.append(vec)
#         else:
#             features.append([0]*136)            # No face → zero vector

# cap.release()

# features = np.array(features)

# csv_path = os.path.join(out_dir, "101_AP1_visual_features_1FPS_136D.csv")
# np.savetxt(csv_path, features, delimiter=",")

# print("Done. Shape:", features.shape)
# print("Saved to:", csv_path)


import cv2
import numpy as np
import dlib
import os

input_dir = r"/mnt/xdrive/sambit/project/data/AP1"
output_dir = r"/mnt/xdrive/sambit/project/data/video/AP1_video_feat_final"
os.makedirs(output_dir, exist_ok=True)

predictor_path = r"/mnt/xdrive/sambit/project/codes/video_features_extraction/models/shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

video_exts = (".mp4", ".avi", ".mov", ".mkv")

for video_name in os.listdir(input_dir):
    if not video_name.lower().endswith(video_exts):
        continue

    video_path = os.path.join(input_dir, video_name)
    print(f"Processing: {video_name}")

    cap = cv2.VideoCapture(video_path)

    features = []
    last_sec = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_sec = int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)

        if current_sec != last_sec:
            last_sec = current_sec

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            if len(faces) > 0:
                landmarks = predictor(gray, faces[0])
                vec = []
                for i in range(68):
                    vec.append(landmarks.part(i).x)
                    vec.append(landmarks.part(i).y)
                features.append(vec)
            else:
                features.append([0] * 136)

    cap.release()

    features = np.array(features)

    out_csv = os.path.join(
        output_dir,
        os.path.splitext(video_name)[0] + "_visual.csv"
    )

    np.savetxt(out_csv, features, delimiter=",")

    print(f"Saved: {out_csv}, Shape: {features.shape}")

print("All videos processed successfully.")

