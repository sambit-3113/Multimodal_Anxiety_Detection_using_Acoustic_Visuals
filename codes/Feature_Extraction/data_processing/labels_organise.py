### Takes labels.csv (SPIN_labels and SR_labels wala folder) and gender from participants_details.xlsx and gives dataset labels as dvlog-style with 70:21:10 train,test,validation


# import pandas as pd
# import os
# import cv2
# import numpy as np

# # ================= PATHS =================
# labels_file = "/mnt/xdrive/sambit/project/labels.csv"
# participants_file = "/mnt/xdrive/sambit/project/Participants_details.xlsx"

# data_05s_dir = "/mnt/xdrive/sambit/project/data/data_0.5s"
# video_dir = "/mnt/xdrive/sambit/project/data/AP1"

# output_dir = "/mnt/xdrive/sambit/project"
# spin_out = os.path.join(output_dir, "label_SPIN.csv")
# sr_out = os.path.join(output_dir, "label_SR.csv")

# # ================= LOAD LABELS =================
# labels_df = pd.read_csv(labels_file)

# # ================= LOAD PARTICIPANTS =================
# participants_df = pd.read_excel(participants_file)

# # Keep only required columns (FIXED)
# participants_df = participants_df[['P_ID', 'Gender']]
# participants_df.columns = ['P_Id', 'gender']

# # ================= MERGE GENDER =================
# df = labels_df.merge(participants_df, on='P_Id', how='left')

# # ================= FILTER BY EXISTING FOLDERS =================
# valid_pids = {
#     name for name in os.listdir(data_05s_dir)
#     if os.path.isdir(os.path.join(data_05s_dir, name))
# }

# df = df[df['P_Id'].astype(str).isin(valid_pids)].copy()

# # ================= VIDEO DURATION =================
# def get_video_duration(video_path):
#     if not os.path.exists(video_path):
#         return 0.0
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#     cap.release()
#     if fps == 0:
#         return 0.0
#     return frames / fps

# df['duration'] = df['P_Id'].astype(str).apply(
#     lambda pid: get_video_duration(os.path.join(video_dir, f"{pid}_AP1.mp4"))
# )

# # ================= 7:1:2 SPLIT =================
# np.random.seed(42)
# pids = df['P_Id'].unique()
# np.random.shuffle(pids)

# n = len(pids)
# train_end = int(0.7 * n)
# valid_end = int(0.8 * n)

# train_pids = pids[:train_end]
# valid_pids = pids[train_end:valid_end]
# test_pids = pids[valid_end:]

# def assign_fold(pid):
#     if pid in train_pids:
#         return "train"
#     elif pid in valid_pids:
#         return "valid"
#     else:
#         return "test"

# df['fold'] = df['P_Id'].apply(assign_fold)

# # ================= CREATE OUTPUT FILES =================
# spin_df = df[['P_Id', 'SPIN_label', 'duration', 'gender', 'fold']].copy()
# spin_df.columns = ['index', 'label', 'duration', 'gender', 'fold']

# sr_df = df[['P_Id', 'SR_label', 'duration', 'gender', 'fold']].copy()
# sr_df.columns = ['index', 'label', 'duration', 'gender', 'fold']

# # ================= SAVE =================
# spin_df.to_csv(spin_out, index=False)
# sr_df.to_csv(sr_out, index=False)

# # ================= LOG =================
# print(" Files created successfully:")
# print(spin_out)
# print(sr_out)
# print("\nFold distribution:")
# print(df['fold'].value_counts())



import pandas as pd
import os
import cv2
import numpy as np

# ================= PATHS =================
labels_file = "/mnt/xdrive/sambit/project/labels.csv"
participants_file = "/mnt/xdrive/sambit/project/Participants_details.xlsx"

data_05s_dir = "/mnt/xdrive/sambit/project/data/data_0.5s"
video_dir = "/mnt/xdrive/sambit/project/data/AP1"

output_dir = "/mnt/xdrive/sambit/project"
spin_out = os.path.join(output_dir, "label_SPIN.csv")
sr_out = os.path.join(output_dir, "label_SR.csv")

# ================= LOAD LABELS =================
labels_df = pd.read_csv(labels_file)

# ================= LOAD PARTICIPANTS =================
participants_df = pd.read_excel(participants_file)

# Keep only required columns
participants_df = participants_df[['P_ID', 'Gender']]
participants_df.columns = ['P_Id', 'gender']

# ---------------- GENDER NORMALIZATION ----------------
participants_df['gender'] = (
    participants_df['gender']
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        'male': 'm',
        'female': 'f'
    })
)

# ================= MERGE GENDER =================
df = labels_df.merge(participants_df, on='P_Id', how='left')

# ================= FILTER BY EXISTING FOLDERS =================
valid_pids = {
    name for name in os.listdir(data_05s_dir)
    if os.path.isdir(os.path.join(data_05s_dir, name))
}

df = df[df['P_Id'].astype(str).isin(valid_pids)].copy()

# ================= VIDEO DURATION =================
def get_video_duration(video_path):
    if not os.path.exists(video_path):
        return 0.0
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps == 0:
        return 0.0
    return frames / fps

df['duration'] = df['P_Id'].astype(str).apply(
    lambda pid: get_video_duration(os.path.join(video_dir, f"{pid}_AP1.mp4"))
)

# ================= 7:1:2 SPLIT =================
np.random.seed(42)
pids = df['P_Id'].unique()
np.random.shuffle(pids)

n = len(pids)
train_end = int(0.7 * n)
valid_end = int(0.8 * n)

train_pids = pids[:train_end]
valid_pids = pids[train_end:valid_end]
test_pids = pids[valid_end:]

def assign_fold(pid):
    if pid in train_pids:
        return "train"
    elif pid in valid_pids:
        return "valid"
    else:
        return "test"

df['fold'] = df['P_Id'].apply(assign_fold)

# ================= CREATE OUTPUT FILES =================
spin_df = df[['P_Id', 'SPIN_label', 'duration', 'gender', 'fold']].copy()
spin_df.columns = ['index', 'label', 'duration', 'gender', 'fold']

sr_df = df[['P_Id', 'SR_label', 'duration', 'gender', 'fold']].copy()
sr_df.columns = ['index', 'label', 'duration', 'gender', 'fold']

# ================= SAVE =================
spin_df.to_csv(spin_out, index=False)
sr_df.to_csv(sr_out, index=False)

# ================= LOG =================
print("Files created successfully:")
print(spin_out)
print(sr_out)
print("\nFold distribution:")
print(df['fold'].value_counts())
