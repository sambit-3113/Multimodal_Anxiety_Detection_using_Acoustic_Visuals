### --------- testing on one file ----------

# import opensmile
# import os

# smile = opensmile.Smile(
#     feature_set=opensmile.FeatureSet.eGeMAPSv02,
#     feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
# )

# llds = smile.process_file(
#     r"C:\Users\sambi\Documents\AP1_audio\101_AP1.wav"
# )

# out_dir = r"C:\Users\sambi\Documents\AP1_audio_feat"
# os.makedirs(out_dir, exist_ok=True)

# out_path = os.path.join(out_dir, "101_AP1.csv")
# llds.to_csv(out_path, index=True)

# print("Saved as:", out_path)





### -------------------- Looping over all the files in the folder --------------------

import opensmile
import os

in_dir = r"C:\Users\sambi\Documents\AP1_audio"
out_dir = r"C:\Users\sambi\Documents\AP1_audio_feat"
os.makedirs(out_dir, exist_ok=True)

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)

for file in os.listdir(in_dir):
    if file.lower().endswith(".wav"):
        in_path = os.path.join(in_dir, file)
        
        llds = smile.process_file(in_path)          # Extract features (SMILE)
        
        base_name = os.path.splitext(file)[0]
        out_name = base_name + "_feat.csv"                      # 105_AP1.wav -> 105_AP1_feat.csv
        out_path = os.path.join(out_dir, out_name)
        
        llds.to_csv(out_path, index=True)
        print("Saved:", out_path)

print("All files processed.")
