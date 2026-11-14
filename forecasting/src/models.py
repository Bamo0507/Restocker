"""Las redes del proyecto: una MLP y un LSTM.

Las dos reciben lo mismo:
- `past`: (lote, días de entrada, rasgos) con la venta normalizada y el calendario de esos días,
- `future`: (lote, 7, rasgos) con el calendario de los días a predecir, que se conoce de antemano,
y devuelven (lote, 7, cuantiles): una predicción por día y por cuantil (1 cuantil = predicción puntual).
"""

import torch
from torch import nn


class MLP(nn.Module):
    """Aplana la ventana y el calendario futuro en una sola fila y la pasa por capas densas."""

    def __init__(self, lookback, past_features, future_features, horizon=7, quantiles=1, hidden=(128, 64), dropout=0.2):
        super().__init__()
        self.horizon, self.quantiles = horizon, quantiles
        layers, size = [], lookback * past_features + horizon * future_features
        for width in hidden:
            layers += [nn.Linear(size, width), nn.ReLU(), nn.Dropout(dropout)]
            size = width
        layers.append(nn.Linear(size, horizon * quantiles))
        self.net = nn.Sequential(*layers)

    def forward(self, past, future):
        x = torch.cat([past.flatten(1), future.flatten(1)], dim=1)
        return self.net(x).view(-1, self.horizon, self.quantiles)


class LSTMForecaster(nn.Module):
    """Lee la ventana día por día y combina su resumen (el hidden state del último día)
    con el calendario de los días a predecir para dar los 7 días de una vez."""

    def __init__(self, past_features, future_features, horizon=7, quantiles=1, hidden=64, layers=2, head=64, dropout=0.2):
        super().__init__()
        self.horizon, self.quantiles = horizon, quantiles
        # El dropout de nn.LSTM solo actúa entre capas, así que con una capa no se usa
        self.lstm = nn.LSTM(past_features, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden + horizon * future_features, head),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head, horizon * quantiles),
        )

    def forward(self, past, future):
        _, (h, _) = self.lstm(past)
        summary = h[-1]  # hidden state del último día, en la última capa
        x = torch.cat([summary, future.flatten(1)], dim=1)
        return self.head(x).view(-1, self.horizon, self.quantiles)
