### ------ data folder structure changed to (as dvlog)------
# /mnt/xdrive/sambit/project/data/data_0.5s/
#    ├── 101/
#    │    ├── 101_acoustic.npy
#    │    └── 101_visual.npy
#    ├── 102/
#    ├── 105/
#    ├── 106/
#    ...

import os
import numpy as np
import pandas as pd

audio_dir = "/mnt/xdrive/sambit/project/data/audio/AP1_audio_feat_final_0.5s"
video_dir = "/mnt/xdrive/sambit/project/data/video/AP1_video_feat_final_0.5s"
out_root  = "/mnt/xdrive/sambit/project/data/data_0.5s"

os.makedirs(out_root, exist_ok=True)        # Creating folder data_0.5s
participants = sorted([f.split("_")[0] for f in os.listdir(audio_dir) if f.endswith(".csv")])       # Get participant IDs from audio CSV files

for pid in participants:
    print(f"Processing participant {pid}")

    pid_folder = os.path.join(out_root, pid)        # Creating participant-wise folders like 101, 102, ... , etc
    os.makedirs(pid_folder, exist_ok=True)

    # -------- Audio --------
    audio_csv = os.path.join(audio_dir, f"{pid}_AP1.csv")
    audio_df = pd.read_csv(audio_csv)
    audio_data = audio_df.iloc[:, 1:].values                        # remove time column (unnecessary)
    audio_out = os.path.join(pid_folder, f"{pid}_acoustic.npy")
    np.save(audio_out, audio_data)

    # -------- Video --------
    video_csv = os.path.join(video_dir, f"{pid}_AP1.csv")
    video_df = pd.read_csv(video_csv)
    video_data = video_df.iloc[:, 1:].values                        # remove time column (unnecessary)
    video_out = os.path.join(pid_folder, f"{pid}_visual.npy")
    np.save(video_out, video_data)

print("Done. data_0.5s folder with participant subfolders is ready.")
