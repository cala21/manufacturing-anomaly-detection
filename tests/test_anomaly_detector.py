import numpy as np
import pytest
from src.training.anomaly_detector import ManufacturingAnomalyDetector


def make_data(n_normal=200, n_anomaly=10):
    rng = np.random.default_rng(42)
    normal = rng.normal(loc=0.0, scale=1.0, size=(n_normal, 5))
    anomalies = rng.normal(loc=10.0, scale=1.0, size=(n_anomaly, 5))
    return np.vstack([normal, anomalies])


def test_fit_returns_self():
    model = ManufacturingAnomalyDetector(contamination=0.05)
    X = make_data()
    result = model.fit(X)
    assert result is model


def test_predict_shape():
    X = make_data()
    model = ManufacturingAnomalyDetector(contamination=0.05).fit(X)
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert set(preds).issubset({1, -1})


def test_anomaly_score_shape():
    X = make_data()
    model = ManufacturingAnomalyDetector(contamination=0.05).fit(X)
    scores = model.anomaly_score(X)
    assert scores.shape == (len(X),)


def test_anomalies_score_lower_than_normal():
    rng = np.random.default_rng(42)
    normal = rng.normal(loc=0.0, scale=0.5, size=(300, 5))
    anomalies = rng.normal(loc=15.0, scale=0.5, size=(10, 5))
    X_train = normal
    model = ManufacturingAnomalyDetector(contamination=0.05).fit(X_train)

    normal_scores = model.anomaly_score(normal).mean()
    anomaly_scores = model.anomaly_score(anomalies).mean()
    assert anomaly_scores < normal_scores, "Anomalies should score lower than normal points"


def test_threshold_set_after_fit():
    model = ManufacturingAnomalyDetector(contamination=0.05)
    assert model.threshold is None
    model.fit(make_data())
    assert model.threshold is not None
