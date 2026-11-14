"""Líneas base sin red neuronal. Trabajan sobre las mismas ventanas que las redes.

Ambas predicen en la escala normalizada (como las redes) y usan solo la entrada de la ventana.
La columna 0 de `past` es la venta normalizada y la columna 1 indica si el local abrió.
"""

import warnings

import numpy as np

from .windows import HORIZON


def _open_sales(windows):
    sales, is_open = windows.past[:, :, 0], windows.past[:, :, 1].astype(bool)
    return np.where(is_open, sales, np.nan)


def seasonal_naive(windows):
    """Igual que la semana pasada: cada día a predecir copia el mismo día de la semana anterior.

    Si ese día estaba cerrado, se busca el mismo día de semanas anteriores dentro de la ventana;
    si no hay ninguno abierto, se usa el promedio de los días abiertos de la ventana.
    """
    sales = _open_sales(windows)
    lookback = sales.shape[1]
    fallback = np.nanmean(sales, axis=1)
    prediction = np.repeat(fallback[:, None], HORIZON, axis=1)
    for h in range(HORIZON):
        filled = np.zeros(len(sales), dtype=bool)
        for back in range(7, lookback + 1, 7):
            column = lookback + h - back
            if column < 0 or column >= lookback:
                continue
            value = sales[:, column]
            use = ~filled & ~np.isnan(value)
            prediction[use, h] = value[use]
            filled |= use
    return prediction


def same_weekday_mean(windows):
    """Promedio de los mismos días de la semana que hay en la ventana (2 con 14 días, 4 con 28).

    Solo cuenta días abiertos; si un día de la semana no tiene ninguno abierto en la ventana,
    se usa el promedio de los días abiertos de toda la ventana.
    """
    sales = _open_sales(windows)
    lookback = sales.shape[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # ventanas sin días abiertos
        fallback = np.nanmean(sales, axis=1)
    prediction = np.empty((len(sales), HORIZON))
    for h in range(HORIZON):
        columns = [lookback + h - back for back in range(7, lookback + 1, 7) if 0 <= lookback + h - back < lookback]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # días de la semana sin ningún día abierto
            mean = np.nanmean(sales[:, columns], axis=1)
        prediction[:, h] = np.where(np.isnan(mean), fallback, mean)
    return prediction
