# engine_predictive_maintenance/hosting/hosting.py

import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ----------------------------
# Validate HF token
# ----------------------------
token = os.getenv("HF_TOKEN_PM")
if token is None:
    raise ValueError("HF_TOKEN_PM environment variable not set")

api = HfApi(token=token)

repo_id = "Vignesh-vigu/Engine-Predictive-Maintenance"
repo_type = "space"

# ----------------------------
# Ensure Space exists
# ----------------------------
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"✅ Space '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"⚠️ Space '{repo_id}' not found. Creating it...")
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        space_sdk="docker",
        private=False
    )
    print(f"✅ Space '{repo_id}' created with Docker SDK.")

# ----------------------------
# Upload deployment folder
# ----------------------------
api.upload_folder(
    folder_path="engine_predictive_maintenance/deployment",
    repo_id=repo_id,
    repo_type=repo_type,
)

print("🚀 Deployment files uploaded to Hugging Face Space successfully.")
