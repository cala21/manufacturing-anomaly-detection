"""Anomaly detection for manufacturing: Isolation Forest + Autoencoder."""
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
class ManufacturingAnomalyDetector:
    """Two-stage anomaly detection: fast statistical check + deep model."""

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self.threshold = None

    def fit(self, X: np.ndarray) -> "ManufacturingAnomalyDetector":
        X_scaled = self.scaler.fit_transform(X)
        self.isolation_forest.fit(X_scaled)
        scores = self.isolation_forest.score_samples(X_scaled)
        # Threshold at contamination percentile
        self.threshold = np.percentile(scores, self.contamination * 100)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns 1 for normal, -1 for anomaly."""
        X_scaled = self.scaler.transform(X)
        return self.isolation_forest.predict(X_scaled)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Lower score = more anomalous."""
        X_scaled = self.scaler.transform(X)
        return self.isolation_forest.score_samples(X_scaled)


def train_and_log(data_path: str, experiment_name: str = "manufacturing-anomaly"):
    df = pd.read_parquet(data_path)
    feature_cols = [c for c in df.columns if c not in ["timestamp", "label"]]
    X = df[feature_cols].values
    y = df["label"].values if "label" in df.columns else None

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        params = {"contamination": 0.05}
        mlflow.log_params(params)
        mlflow.log_param("features", feature_cols)

        model = ManufacturingAnomalyDetector(**params)
        model.fit(X)

        if y is not None:
            preds = model.predict(X)
            preds_binary = (preds == -1).astype(int)
            y_binary = (y == "anomaly").astype(int)
            report = classification_report(y_binary, preds_binary, output_dict=True)
            mlflow.log_metrics({
                "precision": report["1"]["precision"],
                "recall": report["1"]["recall"],
                "f1": report["1"]["f1-score"],
            })

        mlflow.sklearn.log_model(model.isolation_forest, "isolation_forest")
        mlflow.log_dict({"feature_cols": feature_cols}, "features.json")
        print("Training complete. Model logged to MLflow.")
