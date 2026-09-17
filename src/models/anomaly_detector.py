import pandas as pd
import numpy as np
from xgboost import XGBClassifier

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates technical anomaly indicators on time-series window."""
    data = df.copy()
    data["return"] = data["close"].pct_change()
    data["volatility_5"] = data["return"].rolling(window=5).std()
    data["volume_spike"] = data["volume"] / (data["volume"].rolling(window=10).mean() + 1e-6)
    
    # Target label: > 2.5 std devs price return OR > 3x volume spike
    data["is_anomaly"] = (
        (data["return"].abs() > 2.5 * data["return"].std()) | 
        (data["volume_spike"] > 3.0)
    ).astype(int)
    
    data.dropna(inplace=True)
    return data

def train_anomaly_model(df: pd.DataFrame):
    """Fits baseline XGBoost classifier on engineered features."""
    features = ["return", "volatility_5", "volume_spike"]
    X = df[features]
    y = df["is_anomaly"]
    
    model = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1)
    model.fit(X, y)
    return model, features