# import pandas as pd
# import os

# # Path to the labels file
# base_dir = r"/mnt/xdrive/sambit/project/data/data_0.5s"
# input_file = os.path.join(base_dir, "labels.csv")

# # Read CSV
# df = pd.read_csv(input_file)

# # Find mismatches
# mismatch_df = df[df['SR_label'] != df['SPIN_label']]

# # Get P_IDs
# mismatch_pids = mismatch_df['P_Id'].astype(str).tolist()

# print(f"Number of mismatched participants: {len(mismatch_pids)}")
# print("\nP_IDs where SR_label and SPIN_label do not match:")

# for pid in mismatch_pids:
#     print(pid)


# ####----------- for visualize how many anxious, non anxious (in train, test, valid)------
# import pandas as pd

# # Path to CSV
# csv_path = "/mnt/xdrive/sambit/project/data/data_0.5s/labels.csv"

# # Load file
# df = pd.read_csv(csv_path)

# # Function to count classes inside a fold
# def count_classes(dataframe, fold_name):
#     fold_df = dataframe[dataframe["fold"] == fold_name]
    
#     depression_count = (fold_df["label"] == "anxious").sum()
#     normal_count = (fold_df["label"] == "non-anxious").sum()
    
#     print(f"\n{fold_name.capitalize()} Split:")
#     print(f"anxious: {depression_count}")
#     print(f"non-anxious: {normal_count}")

# # Count for each split
# count_classes(df, "train")
# count_classes(df, "test")
# count_classes(df, "valid")
#----------------

# #-------------genderwise counts per split
# import pandas as pd

# # Path to CSV
# csv_path = "/mnt/xdrive/sambit/project/label_SR.csv"

# # Load CSV
# df = pd.read_csv(csv_path)

# # Create pivot table (Gender vs Fold)
# table = pd.pivot_table(
#     df,
#     index="gender",      # Rows
#     columns="fold",      # Columns
#     aggfunc="size",      # Count entries
#     fill_value=0
# )

# # Rename columns for better display
# table = table.rename(columns={
#     "train": "Train",
#     "valid": "Val",
#     "test": "Test"
# })

# # Print table
# print("\nGender-wise Split Distribution:\n")
# print(table)


# #------------ printing the mean, median, max, min no.of rows in the data_0.5s dataset-------- 
import os
import numpy as np
import matplotlib.pyplot as plt

root_dir = r"/mnt/xdrive/sambit/project/dvlog-dataset"

acoustic_rows = []
visual_rows = []

# -------- Load data --------
for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".npy"):
            file_path = os.path.join(root, file)

            try:
                data = np.load(file_path)
                rows = data.shape[0]

                if "acoustic" in file.lower():
                    acoustic_rows.append(rows)
                elif "visual" in file.lower():
                    visual_rows.append(rows)

            except Exception as e:
                print(f"Error loading {file}: {e}")

# -------- Stats --------
def print_stats(name, values):
    if len(values) == 0:
        print(f"\n{name}: No data found")
        return

    values = np.array(values)
    print(f"\n{name} statistics:")
    print(f"Count  : {len(values)}")
    print(f"Mean   : {np.mean(values):.2f}")
    print(f"Median : {np.median(values):.2f}")
    print(f"Max    : {np.max(values)}")
    print(f"Min    : {np.min(values)}")

print_stats("Acoustic (.npy)", acoustic_rows)
print_stats("Visual (.npy)", visual_rows)

# -------- Histogram --------
def plot_histogram(values, title, filename):
    if len(values) == 0:
        print(f"No data available for {title}")
        return

    values = np.array(values)

    # Create bins of size 20 starting from 0
    max_val = int(values.max())
    bins = np.arange(0, max_val + 20, 20)

    plt.figure()
    plt.hist(values, bins=bins)

    plt.xlabel("Number of Rows (bins of 20)")
    plt.ylabel("Number of Participants")
    plt.title(title)

    plt.tight_layout()

    # Save + show
    plt.savefig(filename)
    print(f"{title} saved as {filename}")

    plt.show()

# -------- Plot --------
plot_histogram(acoustic_rows, "Acoustic Data Distribution", "acoustic_hist.png")
plot_histogram(visual_rows, "Visual Data Distribution", "visual_hist.png")

# import pandas as pd

# # Load the CSV files
# df_spin = pd.read_csv('/mnt/xdrive/sambit/project/label_SPIN.csv')
# df_sr = pd.read_csv('/mnt/xdrive/sambit/project/label_SR.csv')

# # Merge on 'index'
# merged = pd.merge(df_spin, df_sr, on='index', suffixes=('_spin', '_sr'))

# # Find mismatches
# mismatches = merged[merged['label_spin'] != merged['label_sr']]

# # Print results
# print(f"Number of mismatched indexes: {len(mismatches)}\n")

# print("Mismatched indexes:")
# print(mismatches['index'].tolist())