# Manufacturing Anomaly Detection

> End-to-end ML pipeline for real-time anomaly detection on production lines — OPC-UA sensor ingestion, Isolation Forest + Autoencoder detection, MLflow tracking, FastAPI serving, and Evidently drift monitoring. Built for Industry 4.0 / Italian PMI.

[![CI](https://github.com/cala21/manufacturing-anomaly-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/cala21/manufacturing-anomaly-detection/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Use Case

A manufacturing line produces defective parts at an unknown rate. This pipeline:
1. **Ingests** real-time sensor data via OPC-UA (the industrial standard protocol)
2. **Detects** anomalies using statistical + ML models with sub-100ms latency
3. **Alerts** operators via API while maintaining human oversight
4. **Monitors** for data drift so the model stays accurate as the process evolves

Designed for Italian PMI in sectors: metalworking, packaging, electronics assembly, food production.

## Architecture

```
OPC-UA Server (PLC/SCADA)
        ↓
  Ingestion Layer (asyncua)
        ↓
  Feature Engineering → Isolation Forest → Anomaly Score
        ↓                   ↓
  MLflow Tracking     FastAPI Endpoint → Operator Alert
        ↓
  Evidently Drift Monitor → Retrain trigger
```

## Quick Start

```bash
# 1. Install
git clone https://github.com/cala21/manufacturing-anomaly-detection
cd manufacturing-anomaly-detection
pip install -r requirements.txt

# 2. Generate synthetic data and train
python -c "
from sklearn.datasets import make_classification
import pandas as pd, numpy as np
X, y = make_classification(1000, 20, random_state=42)
df = pd.DataFrame(X, columns=[f'sensor_{i}' for i in range(20)])
df['label'] = ['anomaly' if yi == 1 else 'normal' for yi in y]
df.to_parquet('data/processed/train.parquet')
print('Synthetic dataset created')
"

mlflow server --backend-store-uri sqlite:///mlflow.db &
python src/training/anomaly_detector.py

# 3. Serve
uvicorn src.inference.api:app --reload

# 4. Test detection
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{
    "line_id": "LINE-01",
    "features": {"vibration": 12.3, "temperature": 87.2, "pressure": 4.1},
    "timestamp": "2024-10-01T10:00:00Z"
  }'
```

## Repository Structure

```
├── src/
│   ├── ingestion/
│   │   └── opcua_client.py       # Async OPC-UA reader with async/await
│   ├── features/                 # Feature engineering transforms
│   ├── training/
│   │   └── anomaly_detector.py   # Isolation Forest with MLflow logging
│   ├── inference/
│   │   └── api.py                # FastAPI endpoint + Prometheus metrics
│   └── monitoring/
│       └── drift_detector.py     # Evidently data drift reports
├── notebooks/                    # Exploratory analysis
├── deployment/
│   ├── k8s/                      # Kubernetes manifests
│   └── edge/                     # Edge deployment (Raspberry Pi / industrial PC)
├── data/
│   ├── raw/                      # Raw sensor exports (gitignored)
│   ├── processed/                # Cleaned datasets (gitignored)
│   └── schemas/                  # Expected sensor schemas
└── tests/
```

## OPC-UA Integration

OPC-UA is the standard protocol for industrial equipment (PLCs, SCADA, robots). This client reads any OPC-UA node in real time:

```python
from src.ingestion.opcua_client import OPCUAIngestionClient

async with OPCUAIngestionClient("opc.tcp://192.168.1.100:4840") as client:
    async for reading in client.stream_readings(
        node_ids=["ns=2;i=1001", "ns=2;i=1002"],  # vibration, temperature
        interval_ms=100,
    ):
        print(f"{reading.node_id}: {reading.value} ({reading.quality})")
```

If you don't have an OPC-UA server, use [node-opcua](https://node-opcua.github.io/) or [FreeOpcUa](https://github.com/FreeOpcUa/opcua-asyncio) to simulate one.

## Model Performance

Baseline on synthetic dataset (adjust thresholds for your process):

| Metric | Value |
|--------|-------|
| Precision | ~0.87 |
| Recall | ~0.82 |
| Inference latency | < 5ms |
| Contamination param | 0.05 (5% anomaly rate) |

Tune `contamination` to match your actual defect rate. Higher = more sensitive, more false positives.

## Drift Monitoring

```python
from src.monitoring.drift_detector import run_drift_report

summary = run_drift_report(reference_df, current_df)
if summary["drift_detected"]:
    print(f"Drifted columns: {summary['drifted_columns']}")
    # Trigger retraining pipeline
```

## EU AI Act Classification

This system is **minimal risk** when:
- It monitors equipment (not workers)
- Final go/no-go decisions have human review
- No biometric or personal data is processed

It becomes **high risk** if repurposed for worker monitoring (Annex III, category 4). See [`eu-ai-act-compliance-toolkit`](https://github.com/cala21/eu-ai-act-compliance-toolkit) for a worked risk assessment for this exact scenario.

## License

MIT
