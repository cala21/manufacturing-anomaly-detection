# Reference Datasets

## SECOM Semiconductor Manufacturing
- Source: UCI ML Repository
- URL: https://archive.ics.uci.edu/dataset/179/secom
- 1567 samples, 590 features, ~93% normal / ~7% defective
- Replace synthetic data in training script with this for realistic evaluation

## NASA CMAPSS Turbofan Degradation
- Source: NASA Prognostics Center
- URL: https://data.nasa.gov/dataset/CMAPSS-Jet-Engine-Simulated-Data
- Time-series sensor readings from jet engine runs to failure
- Good for predictive maintenance scenarios

## MIMII (Malfunctioning Industrial Machine Investigation and Inspection)
- Source: Dcase Challenge
- URL: https://zenodo.org/record/3384388
- Acoustic anomaly detection for industrial machines
- Adapt src/ingestion/ for audio input

## Generating Synthetic Data (quick start)
```python
python -c "
from sklearn.datasets import make_classification
import pandas as pd
X, y = make_classification(1000, 20, weights=[0.95, 0.05], random_state=42)
df = pd.DataFrame(X, columns=[f'sensor_{i}' for i in range(20)])
df['label'] = ['anomaly' if yi == 1 else 'normal' for yi in y]
df.to_parquet('data/processed/train.parquet')
"
```
