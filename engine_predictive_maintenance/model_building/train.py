# ----------------------------
# Production Training Script
# ----------------------------

import os
import pandas as pd
import joblib
import mlflow
import xgboost as xgb

from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ----------------------------
# Validate HF Token
# ----------------------------
token = os.getenv("HF_TOKEN_PM")
if token is None:
    raise ValueError("HF_TOKEN_PM environment variable not set")

api = HfApi(token=token)

mlflow.set_experiment("engine_predictive_maintenance_training")

# ----------------------------
# Load processed data
# ----------------------------
X_train = pd.read_csv(
    "hf://datasets/Vignesh-vigu/Engine-Predictive-Maintenance/Xtrain.csv"
)
X_test = pd.read_csv(
    "hf://datasets/Vignesh-vigu/Engine-Predictive-Maintenance/Xtest.csv"
)
y_train = pd.read_csv(
    "hf://datasets/Vignesh-vigu/Engine-Predictive-Maintenance/ytrain.csv"
).squeeze()
y_test = pd.read_csv(
    "hf://datasets/Vignesh-vigu/Engine-Predictive-Maintenance/ytest.csv"
).squeeze()

print("✅ Processed data loaded")

# ----------------------------
# Feature list (NORMALIZED)
# ----------------------------
numeric_features = [
    "engine_rpm",
    "lub_oil_pressure",
    "fuel_pressure",
    "coolant_pressure",
    "lub_oil_temp",
    "coolant_temp",
]

# ----------------------------
# Handle class imbalance
# ----------------------------
class_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

# ----------------------------
# Pipeline
# ----------------------------
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features)
)

xgb_model = xgb.XGBClassifier(
    n_estimators=75,
    max_depth=3,
    learning_rate=0.1,
    colsample_bytree=0.6,
    reg_lambda=0.6,
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric="logloss"
)

model = make_pipeline(preprocessor, xgb_model)

# ----------------------------
# Train & Evaluate
# ----------------------------
with mlflow.start_run():
    model.fit(X_train, y_train)

    threshold = 0.45
    y_pred_test = (
        model.predict_proba(X_test)[:, 1] >= threshold
    ).astype(int)

    report = classification_report(
        y_test, y_pred_test, output_dict=True
    )

    mlflow.log_metrics({
        "accuracy": report["accuracy"],
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1_score": report["1"]["f1-score"]
    })

    # Save model
    model_path = "engine_predictive_maintenance_model.joblib"
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)

# ----------------------------
# Upload model to HF Hub
# ----------------------------
repo_id = "Vignesh-vigu/Engine-Predictive-Maintenance"
repo_type = "model"

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=model_path,
    repo_id=repo_id,
    repo_type=repo_type,
)

print("✅ Training complete. Model uploaded.")

