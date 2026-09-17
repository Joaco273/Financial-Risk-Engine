from src.data.ingestion import fetch_historical_ticks
from src.models.anomaly_detector import engineer_features, train_anomaly_model

print("[1/3] Fetching market data...")
raw_df = fetch_historical_ticks("SPY", period="5d", interval="1m")

print("[2/3] Engineering features & training XGBoost model...")
processed_df = engineer_features(raw_df)
model, feature_cols = train_anomaly_model(processed_df)

print("[3/3] Testing real-time anomaly detection output...")
recent_sample = processed_df.tail(10)
predictions = model.predict(recent_sample[feature_cols])

for idx, pred in zip(recent_sample.index, predictions):
    status = "🚨 ANOMALY" if pred == 1 else "OK"
    print(f"[{idx}] Status: {status}")