"""Exporta los pesos del checkpoint de PyTorch a un .npz para servir el modelo sin torch.

El servicio en Vercel no puede llevar torch (pesa ~700 MB y el límite es 500 MB), así que
src/numpy_model.py repite el forward de LSTMForecaster con NumPy a partir de estos pesos.

Uso (desde forecasting/, con el entorno de los notebooks):
    .venv/bin/python scripts/export_weights.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "models" / "05_lstm_cuantiles.pt"
OUTPUT = CHECKPOINT.with_suffix(".npz")

state = torch.load(CHECKPOINT, map_location="cpu")
weights = {name.replace(".", "__"): tensor.numpy() for name, tensor in state.items()}
np.savez(OUTPUT, **weights)

for name, array in weights.items():
    print(f"{name:24s} {tuple(array.shape)}")
print(f"\nGuardado en {OUTPUT.relative_to(ROOT)} ({OUTPUT.stat().st_size / 1024:.0f} KB)", file=sys.stderr)
