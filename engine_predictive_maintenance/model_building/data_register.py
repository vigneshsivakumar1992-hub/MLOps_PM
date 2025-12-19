
import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

repo_id = "Vignesh-vigu/Engine-Predictive-Maintenance"
repo_type = "dataset"

token = os.getenv("HF_TOKEN_PM")
if token is None:
    raise ValueError("HF_TOKEN_PM not set")

api = HfApi(token=token)

# Check if dataset exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"✅ Dataset '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"⚠️ Dataset '{repo_id}' not found. Creating it...")
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False
    )
    print(f"✅ Dataset '{repo_id}' created.")

# Upload data folder
api.upload_folder(
    folder_path="engine_predictive_maintenance/data",
    repo_id=repo_id,
    repo_type=repo_type,
)

print("🚀 Dataset uploaded successfully.")
