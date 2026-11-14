"""Recorta las series del dataset unificado en ejemplos de entrenamiento (ventana deslizante).

Cada ejemplo tiene:
- una entrada con los `lookback` días anteriores (ventas normalizadas y variables de calendario),
- las variables de los `horizon` días a predecir que se conocen de antemano (días abiertos y feriados),
- la salida: las ventas de esos días, normalizadas con la misma escala de la entrada.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

HORIZON = 7
CALENDAR_COLUMNS = ["is_open", "is_holiday", "is_holiday_next2days"]


@dataclass
class Windows:
    past: np.ndarray    # (n, lookback, rasgos): venta normalizada + calendario de los días de entrada
    future: np.ndarray  # (n, horizon, rasgos): calendario de los días a predecir
    target: np.ndarray  # (n, horizon): venta normalizada de los días a predecir
    mask: np.ndarray    # (n, horizon): 1 si el local abrió ese día (solo esos cuentan para el error)
    scale: np.ndarray   # (n,): escala de la ventana, para volver a unidades
    meta: pd.DataFrame  # una fila por ejemplo: dataset, local, producto, primer día predicho y split

    def __len__(self):
        return len(self.target)


def day_of_week_features(dates):
    """Día de la semana como seno y coseno, para que domingo y lunes queden cerca."""
    angle = 2 * np.pi * dates.dt.day_of_week.to_numpy() / 7
    return np.column_stack([np.sin(angle), np.cos(angle)])


def window_scale(sales, is_open):
    """Escala de una ventana: 1 + promedio de las ventas de sus días abiertos (como DeepAR).

    El 1 evita dividir entre cero cuando un producto no vendió nada en toda la ventana.
    """
    open_days = is_open.astype(bool)
    return 1.0 + sales[open_days].mean()


def make_windows(df, lookback, splits, horizon=HORIZON, use_day_of_week=False, first_offset=None):
    """Arma los ejemplos cuyos días a predecir pertenecen todos a alguno de `splits`.

    La entrada puede venir de una parte anterior (por ejemplo, una ventana de validación
    toma su entrada del periodo de entrenamiento), porque es pasado y no hay fuga.

    `first_offset` fija desde qué día de cada serie se empieza a predecir. Para comparar
    ventanas de 14 y 28 días sobre los mismos días, ambas se evalúan con `first_offset=28`.
    """
    first_offset = max(lookback, first_offset or lookback)
    past, future, target, mask, scale, meta = [], [], [], [], [], []

    for (dataset, store, product), series in df.groupby(["dataset", "store_id", "product_id"], sort=True):
        series = series.sort_values("date")
        days = (series["date"].diff().dt.days.dropna() != 1).sum()
        assert days == 0, f"La serie {store}/{product} tiene días faltantes"

        sales = series["sales"].to_numpy(dtype=float)
        calendar = series[CALENDAR_COLUMNS].to_numpy(dtype=float)
        if use_day_of_week:
            calendar = np.column_stack([calendar, day_of_week_features(series["date"])])
        split = series["split"].to_numpy()
        is_open = series["is_open"].to_numpy()
        dates = series["date"].to_numpy()

        for start in range(first_offset, len(series) - horizon + 1):
            inputs, outputs = slice(start - lookback, start), slice(start, start + horizon)
            if not np.isin(split[outputs], splits).all() or len(set(split[outputs])) > 1:
                continue
            if not is_open[inputs].any() or not is_open[outputs].any():
                continue

            s = window_scale(sales[inputs], is_open[inputs])
            past.append(np.column_stack([sales[inputs] / s, calendar[inputs]]))
            future.append(calendar[outputs])
            target.append(sales[outputs] / s)
            mask.append(is_open[outputs].astype(float))
            scale.append(s)
            meta.append((dataset, store, product, dates[start], split[start]))

    return Windows(
        past=np.asarray(past, dtype=np.float32),
        future=np.asarray(future, dtype=np.float32),
        target=np.asarray(target, dtype=np.float32),
        mask=np.asarray(mask, dtype=np.float32),
        scale=np.asarray(scale, dtype=np.float32),
        meta=pd.DataFrame(meta, columns=["dataset", "store_id", "product_id", "first_date", "split"]),
    )
