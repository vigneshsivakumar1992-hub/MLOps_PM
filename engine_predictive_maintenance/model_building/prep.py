
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# Validate HF token
token = os.getenv("HF_TOKEN_PM")
if token is None:
    raise ValueError("HF_TOKEN_PM environment variable not set")

api = HfApi(token=token)

# Load dataset from Hugging Face
DATASET_PATH = "hf://datasets/Vignesh-vigu/Engine-Predictive-Maintenance/engine_data.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# 🔴 CRITICAL: Normalize column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# Target column
target_col = "engine_condition"

# Split features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save processed data
os.makedirs("processed", exist_ok=True)

Xtrain.to_csv("processed/Xtrain.csv", index=False)
Xtest.to_csv("processed/Xtest.csv", index=False)
ytrain.to_csv("processed/ytrain.csv", index=False)
ytest.to_csv("processed/ytest.csv", index=False)

# Upload processed files to HF dataset
for file_path in [
    "processed/Xtrain.csv",
    "processed/Xtest.csv",
    "processed/ytrain.csv",
    "processed/ytest.csv"
]:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=os.path.basename(file_path),
        repo_id="Vignesh-vigu/Engine-Predictive-Maintenance",
        repo_type="dataset",
    )

print("Data preparation and upload completed successfully.")
