import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load data
DATA_PATH = os.path.join(BASE_DIR, "amazon_sales_clean.csv")
data = pd.read_csv(DATA_PATH)

X = data.drop(columns=["label_profit"])
y = (data["label_profit"] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
acc = accuracy_score(y_test, model.predict(X_test))
print(f"Training selesai | Accuracy: {acc}")

# Save model sebagai file .pkl
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
joblib.dump(model, MODEL_PATH)
print(f"Model tersimpan di {MODEL_PATH}")

