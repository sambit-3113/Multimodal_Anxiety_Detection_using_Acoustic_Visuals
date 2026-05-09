# ### ----- frame-wise feature extraction and averaging (0.5sec bins) testcode for 1 file -----

# import cv2
# import numpy as np
# import dlib
# import os

# video_path = r"/mnt/xdrive/sambit/project/data/AP1/101_AP1.mp4"
# output_dir = r"/mnt/xdrive/sambit/project/data/video/AP1_vdo_feat_demo"
# os.makedirs(output_dir, exist_ok=True)

# predictor_path = r"/mnt/xdrive/sambit/project/codes/video_features_extraction/models/shape_predictor_68_face_landmarks.dat"

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(predictor_path)

# cap = cv2.VideoCapture(video_path)

# bin_size = 0.5  # seconds
# current_bin_start = 0.0
# bin_features = []
# final_features = []

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray)

#     if len(faces) > 0:
#         landmarks = predictor(gray, faces[0])
#         vec = []
#         for i in range(68):
#             vec.append(landmarks.part(i).x)
#             vec.append(landmarks.part(i).y)
#     else:
#         vec = [0] * 136

#     # Check if frame belongs to current bin
#     if timestamp < current_bin_start + bin_size:
#         bin_features.append(vec)
#     else:
#         # Average current bin
#         if len(bin_features) > 0:
#             mean_vec = np.mean(bin_features, axis=0)
#         else:
#             mean_vec = np.zeros(136)

#         final_features.append([current_bin_start] + mean_vec.tolist())

#         # Move to next bin
#         current_bin_start += bin_size
#         bin_features = [vec]

# cap.release()

# # Handle last bin
# if len(bin_features) > 0:
#     mean_vec = np.mean(bin_features, axis=0)
#     final_features.append([current_bin_start] + mean_vec.tolist())

# final_features = np.array(final_features)

# # Column names
# columns = ["time_sec"]
# for i in range(68):
#     columns.append(f"x{i}")
#     columns.append(f"y{i}")

# out_csv = os.path.join(output_dir, "101_AP1.csv")
# header = ",".join(columns)

# np.savetxt(out_csv, final_features, delimiter=",", header=header, comments="")

# print("Saved:", out_csv)
# print("Shape:", final_features.shape)


### ----- frame-wise feature extraction and averaging (0.5sec bins) for all files in the folder -----
import cv2
import numpy as np
import dlib
import os

input_dir  = r"/mnt/xdrive/sambit/project/data/AP1"
output_dir = r"/mnt/xdrive/sambit/project/data/video/AP1_video_feat_final_0.5s"
os.makedirs(output_dir, exist_ok=True)

predictor_path = r"/mnt/xdrive/sambit/project/codes/video_features_extraction/models/shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

bin_size = 0.5  # seconds
video_exts = (".mp4", ".avi", ".mov", ".mkv")

for video_name in os.listdir(input_dir):
    if not video_name.lower().endswith(video_exts):
        continue

    print("Processing:", video_name)

    video_path = os.path.join(input_dir, video_name)
    cap = cv2.VideoCapture(video_path)

    current_bin_start = 0.0
    bin_features = []
    final_features = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) > 0:
            landmarks = predictor(gray, faces[0])
            vec = []
            for i in range(68):
                vec.append(landmarks.part(i).x)
                vec.append(landmarks.part(i).y)
        else:
            vec = [0] * 136

        while timestamp >= current_bin_start + bin_size:
            if len(bin_features) > 0:
                mean_vec = np.mean(bin_features, axis=0)
            else:
                mean_vec = np.zeros(136)

            final_features.append([current_bin_start] + mean_vec.tolist())
            current_bin_start += bin_size
            bin_features = []

        bin_features.append(vec)

    cap.release()

    # last bin
    if len(bin_features) > 0:
        mean_vec = np.mean(bin_features, axis=0)
        final_features.append([current_bin_start] + mean_vec.tolist())

    final_features = np.array(final_features)

    # Column names
    columns = ["time_sec"]
    for i in range(68):
        columns.append(f"x{i}")
        columns.append(f"y{i}")
    header = ",".join(columns)

    # Output name: 101_AP1.csv, 102_AP1.csv, ...
    base_name = os.path.splitext(video_name)[0]
    out_csv = os.path.join(output_dir, base_name + ".csv")

    np.savetxt(out_csv, final_features, delimiter=",", header=header, comments="")

    print("Saved:", out_csv, " Shape:", final_features.shape)

print("All videos processed with 0.5-second averaged visual features.")

