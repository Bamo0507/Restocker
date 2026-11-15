"""El forward de LSTMForecaster (src/models.py) en NumPy, para servir el modelo sin torch.

Usa los pesos que exporta scripts/export_weights.py y repite las mismas cuentas que nn.LSTM:
por cada día y cada capa, las compuertas de entrada (i), olvido (f), candidato (g) y salida (o),

    i, f, g, o = W_ih x + b_ih + W_hh h + b_hh      (en ese orden, como PyTorch)
    c = σ(f) c + σ(i) tanh(g)
    h = σ(o) tanh(c)

y luego la cabeza Linear → ReLU → Linear sobre [h del último día, calendario de los 7 días].
En inferencia el dropout no actúa, así que no aparece. scripts/check_numpy_model.py verifica
que da lo mismo que la red de PyTorch.
"""

from pathlib import Path

import numpy as np


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))  # misma función, sin desbordes con x muy negativo


class NumpyForecaster:
    def __init__(self, weights_path: Path, horizon: int = 7, quantiles: int = 9):
        weights = np.load(weights_path)
        self.w = {name.replace("__", "."): weights[name].astype(np.float64) for name in weights.files}
        self.layers = sum(1 for name in self.w if name.startswith("lstm.weight_ih_l"))
        self.hidden = self.w["lstm.weight_hh_l0"].shape[1]
        self.horizon, self.quantiles = horizon, quantiles

    def _lstm_layer(self, x, layer):
        """x: (lote, días, entradas) → h de cada día (lote, días, hidden)."""
        w_ih, w_hh = self.w[f"lstm.weight_ih_l{layer}"], self.w[f"lstm.weight_hh_l{layer}"]
        bias = self.w[f"lstm.bias_ih_l{layer}"] + self.w[f"lstm.bias_hh_l{layer}"]
        batch, days, _ = x.shape
        h = np.zeros((batch, self.hidden))
        c = np.zeros((batch, self.hidden))
        outputs = np.empty((batch, days, self.hidden))
        for t in range(days):
            gates = x[:, t] @ w_ih.T + h @ w_hh.T + bias
            i, f, g, o = np.split(gates, 4, axis=1)
            c = _sigmoid(f) * c + _sigmoid(i) * np.tanh(g)
            h = _sigmoid(o) * np.tanh(c)
            outputs[:, t] = h
        return outputs

    def __call__(self, past, future):
        """past: (lote, 28, 4) y future: (lote, 7, 3) → escalera escalada (lote, 7, cuantiles)."""
        x = np.asarray(past, dtype=np.float64)
        for layer in range(self.layers):
            x = self._lstm_layer(x, layer)
        summary = x[:, -1]  # h del último día en la última capa
        features = np.concatenate([summary, np.asarray(future, dtype=np.float64).reshape(len(summary), -1)], axis=1)
        hidden = np.maximum(features @ self.w["head.0.weight"].T + self.w["head.0.bias"], 0.0)
        out = hidden @ self.w["head.3.weight"].T + self.w["head.3.bias"]
        return out.reshape(-1, self.horizon, self.quantiles)
