import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

class AnomalyDetector:
    def __init__(self, contamination=0.05):
        self.contamination = contamination
        self.scaler = RobustScaler()
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=150
        )
        self.threshold = None

    def fit(self, X, y=None):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        scores = self.model.score_samples(X_scaled)
        self.threshold = np.percentile(scores, 5)
        print(f"🧹 Модель Isolation Forest обучена")
        print(f"🎯 Порог аномалии: {self.threshold:.4f}")
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        scores = self.model.score_samples(X_scaled)
        return (scores > self.threshold).astype(int)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        scores = self.model.score_samples(X_scaled)
        proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        return proba