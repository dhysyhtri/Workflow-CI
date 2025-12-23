import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# MLflow tracking DagsHub
mlflow.set_tracking_uri("https://dagshub.com/dhysyhtri/amazon-sales-mlflow.mlflow")
mlflow.set_experiment("amazon-sales-ci")

# Load dataset
DATA_PATH = os.path.join(BASE_DIR, "amazon_sales_clean.csv")
data = pd.read_csv(DATA_PATH)

X = data.drop(columns=["label_profit"])
y = (data["label_profit"] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run() as run:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Training selesai | Accuracy: {acc}")

    # Log metric & parameter
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_metric("accuracy", acc)

    # Log model ke MLflow (artifact_path='model')
    mlflow.sklearn.log_model(model, artifact_path="model")

    print(f"RUN_ID: {run.info.run_id}")
