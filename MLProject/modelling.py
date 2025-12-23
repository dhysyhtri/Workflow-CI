import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EXPERIMENT_NAME = "amazon-sales-ci"
mlflow.set_experiment(EXPERIMENT_NAME)

# Load data
DATA_PATH = os.path.join(BASE_DIR, "amazon_sales_clean.csv")
data = pd.read_csv(DATA_PATH)

X = data.drop(columns=["label_profit"])
y = (data["label_profit"] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

    print(f"Training selesai | Accuracy: {acc}")
