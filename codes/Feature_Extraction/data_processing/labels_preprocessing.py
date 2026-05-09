# ## Takes the labels.xlsx file extracts the AS1 columns then reverts the AS1_2 and sums the value as SR_labels


# import pandas as pd
# import os

# # Paths
# input_file = r"/mnt/xdrive/sambit/project/labels.xlsx"
# base_dir = r"/mnt/xdrive/sambit/project/data/data_0.5s"
# output_file = os.path.join(base_dir, "labels_cleaned.csv")

# # Columns to keep
# cols = ['P_Id', 'SPIN Score', 'AS1_1', 'AS1_2', 'AS1_3', 'AS1_4', 'AS1_5']
# pid_col = 'P_Id'

# # Read Excel
# df = pd.read_excel(input_file)

# # Keep only required columns
# df_selected = df[cols]

# # Original P_IDs
# original_pids = set(df_selected[pid_col].dropna().astype(str))

# # Drop rows with NaN in these columns
# df_clean = df_selected.dropna()

# # Clean P_IDs
# clean_pids = set(df_clean[pid_col].astype(str))
# unique_pids = len(clean_pids)

# # Save cleaned CSV
# os.makedirs(base_dir, exist_ok=True)
# df_clean.to_csv(output_file, index=False)

# # P_IDs removed due to NaNs
# removed_pids = original_pids - clean_pids

# # Print results
# print(f"Number of P_IDs after removing NaN rows: {unique_pids}")
# print(f"Cleaned labels CSV saved at: {output_file}")

# print("\nP_IDs removed due to NaN values in selected columns:")
# for pid in sorted(removed_pids):
#     print(pid)


import pandas as pd
import os

# Paths
input_file = r"/mnt/xdrive/sambit/project/labels.xlsx"
base_dir = r"/mnt/xdrive/sambit/project/data/data_0.5s"
output_file = os.path.join(base_dir, "labels.csv")

# Columns to keep
cols = ['P_Id', 'SPIN Score', 'AS1_1', 'AS1_2', 'AS1_3', 'AS1_4', 'AS1_5']
pid_col = 'P_Id'

# Read Excel
df = pd.read_excel(input_file)

# Keep only required columns
df_selected = df[cols]

# Original P_IDs
original_pids = set(df_selected[pid_col].dropna().astype(str))

# Drop rows with NaN in selected columns
df_clean = df_selected.dropna()

# Removed P_IDs
clean_pids = set(df_clean[pid_col].astype(str))
removed_pids = original_pids - clean_pids

# ---- New Columns ----

# Inverted AS1_2
df_clean['AS1_2i'] = 6 - df_clean['AS1_2']

# AS1 Sum
df_clean['AS1_Sum'] = (
    df_clean['AS1_1'] +
    df_clean['AS1_2i'] +
    df_clean['AS1_3'] +
    df_clean['AS1_4'] +
    df_clean['AS1_5']
)

# SR label
df_clean['SR_label'] = df_clean['AS1_Sum'].apply(
    lambda x: 'anxious' if x > 14 else 'non-anxious'
)

# SPIN label
df_clean['SPIN_label'] = df_clean['SPIN Score'].apply(
    lambda x: 'anxious' if x > 25 else 'non-anxious'
)

# Save CSV
os.makedirs(base_dir, exist_ok=True)
df_clean.to_csv(output_file, index=False)

# Print results
print(f"Cleaned labels saved at: {output_file}")
print(f"Number of participants after cleaning: {len(df_clean)}")

print("\nP_IDs removed due to NaN values:")
for pid in sorted(removed_pids):
    print(pid)
