"""Verifica que el forward en NumPy (src/numpy_model.py) da lo mismo que la red de PyTorch.

Compara la salida de ambos en todas las ventanas de 28 días de los tres datasets (train,
validación y prueba). Si la diferencia máxima pasa de 1e-4, falla.

Uso (desde forecasting/, con el entorno de los notebooks):
    .venv/bin/python scripts/check_numpy_model.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference import QUANTILES, load_model  # noqa: E402
from src.models import LSTMForecaster  # noqa: E402
from src.windows import make_windows  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

df = pd.read_parquet(ROOT / "data" / "processed" / "unified.parquet")
windows = make_windows(df, 28, sorted(df["split"].unique()), first_offset=28)

net = LSTMForecaster(past_features=4, future_features=3, quantiles=len(QUANTILES), hidden=32, layers=2)
net.load_state_dict(torch.load(ROOT / "models" / "05_lstm_cuantiles.pt", map_location="cpu"))
net.eval()
with torch.no_grad():
    expected = net(torch.from_numpy(windows.past), torch.from_numpy(windows.future)).numpy()

got = load_model()(windows.past, windows.future)
diff = np.abs(got - expected)
print(f"Ventanas comparadas: {len(windows):,}")
print(f"Diferencia máxima:   {diff.max():.2e}")
print(f"Diferencia media:    {diff.mean():.2e}")
sys.exit(0 if diff.max() < 1e-4 else 1)
