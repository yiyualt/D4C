import pandas as pd
import os

data_folder = './data'
folder_pairs = [
    ("AM_Electronics", "AM_CDs"),
    ("AM_Movies", "AM_CDs"),
    ("AM_CDs", "AM_Electronics"),
    ("AM_Movies", "AM_Electronics"),
    ("AM_CDs", "AM_Movies"),
    ("AM_Electronics", "AM_Movies"),
    ("Yelp", "TripAdvisor"),
    ("TripAdvisor", "Yelp")
]

columns = ["user", "item", "rating", "review", "explanation", "user_idx", "item_idx"]

def merge_data(source, target, output_folder):
    # Read the train and valid CSV files
    source_train = os.path.join(data_folder, source, "train.csv")
    target_train = os.path.join(data_folder, target, "train.csv")
    source_valid = os.path.join(data_folder, source, "valid.csv")
    target_valid = os.path.join(data_folder, target, "valid.csv")

    source_train_data = pd.read_csv(source_train)
    target_train_data = pd.read_csv(target_train)
    source_valid_data = pd.read_csv(source_valid)
    target_valid_data = pd.read_csv(target_valid)

    # Find max user and item indices in the target domain
    max_user_idx = target_train_data["user_idx"].max()
    max_item_idx = target_train_data["item_idx"].max()

    # Increment user and item indices in the source domain
    source_train_data["user_idx"] += max_user_idx + 1
    source_train_data["item_idx"] += max_item_idx + 1
    source_valid_data["user_idx"] += max_user_idx + 1
    source_valid_data["item_idx"] += max_item_idx + 1

    # Add the domain column
    source_train_data["domain"] = "auxiliary"
    target_train_data["domain"] = "target"
    source_valid_data["domain"] = "auxiliary"
    target_valid_data["domain"] = "target"

    # Merge source and target data
    merged_train_data = pd.concat([source_train_data[columns + ["domain"]], target_train_data[columns + ["domain"]]], ignore_index=True)
    merged_valid_data = pd.concat([source_valid_data[columns + ["domain"]], target_valid_data[columns + ["domain"]]], ignore_index=True)

    # Save the merged files
    os.makedirs(os.path.join(data_folder, output_folder), exist_ok=True)
    merged_train_file = os.path.join(data_folder, output_folder, "aug_train.csv")
    merged_valid_file = os.path.join(data_folder, output_folder, "aug_valid.csv")
    merged_train_data.to_csv(merged_train_file, index=False)
    merged_valid_data.to_csv(merged_valid_file, index=False)

    print(f"Merged train data saved to {merged_train_file}")
    print(f"Merged valid data saved to {merged_valid_file}")

# Loop through folder pairs and merge data
for idx, (source, target) in enumerate(folder_pairs, 1):
    merge_data(source, target, str(idx))
