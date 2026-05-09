### --------- testing on one file ---------

# import pandas as pd

# # Load the LLD CSV
# csv_path = r"C:\Users\sambi\Documents\AP1_audio_feat\101_AP1.csv"
# df = pd.read_csv(csv_path)

# # Convert start time to seconds
# df['start_sec'] = pd.to_timedelta(df['start']).dt.total_seconds()

# # Create 1-second bins: 0–1, 1–2, 2–3, ...
# df['sec_bin'] = df['start_sec'].astype(int)

# # Drop non-feature columns
# feature_cols = df.columns.drop(['file', 'start', 'end', 'start_sec', 'sec_bin'])

# # Average LLDs in each 1-second window
# df_1s = df.groupby('sec_bin')[feature_cols].mean()

# # Save
# out_path = r"C:\Users\sambi\Documents\AP1_audio_feat\101_AP1_1s_avg.csv"
# df_1s.to_csv(out_path)

# print("1-second averaged features saved at:", out_path)




### --------- Looping over all files in the folder -----------

import pandas as pd
import os

in_dir = r"C:\Users\sambi\Documents\AP1_audio_feat"
out_dir = r"C:\Users\sambi\Documents\AP1_audio_feat_final"
os.makedirs(out_dir, exist_ok=True)

for file in os.listdir(in_dir):
    if file.endswith("_feat.csv"):
        csv_path = os.path.join(in_dir, file)
        df = pd.read_csv(csv_path)

        df['start_sec'] = pd.to_timedelta(df['start']).dt.total_seconds()       # Converting start time to seconds
        df['sec_bin'] = df['start_sec'].astype(int)                             # Create 1-second bins

        feature_cols = df.columns.drop(['file', 'start', 'end', 'start_sec', 'sec_bin'])        # Drop non-feature columns

        df_1s = df.groupby('sec_bin')[feature_cols].mean()                      # Average LLDs in each 1-second window
        base_name = file.replace("_feat.csv", ".csv")                           # Output name: 153_AP1_feat.csv -> 153_AP1.csv
        out_path = os.path.join(out_dir, base_name)

        df_1s.to_csv(out_path)
        print("Saved:", out_path)

print("All files processed and saved in:", out_dir)
