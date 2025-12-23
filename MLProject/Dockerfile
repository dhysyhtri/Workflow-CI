# Dockerfile untuk model Amazon Sales MLflow
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Salin model.pkl ke container
COPY model.pkl .

# Install dependencies langsung
RUN pip install --no-cache-dir pandas numpy scikit-learn joblib

# Command default untuk test model saat container jalan
CMD ["python", "-c", "import joblib; model=joblib.load('model.pkl'); print('Model siap digunakan')"]
