"""Modelo de valor (newsvendor): traduce las predicciones a cuánto producir y cuánto cuesta equivocarse.

Para cada producto y día abierto:
- si se produjo menos de lo que se vendió, cada unidad que faltó cuesta Cu = precio − costo (margen perdido);
- si se produjo de más, cada unidad que sobró cuesta Co = costo − valor de rescate (lo que se tira).
El cuantil óptimo a producir es q* = Cu / (Cu + Co).
"""

import numpy as np
import pandas as pd


def load_assumptions(path, cost_factor=1.0, price_factor=1.0, salvage_pct=0.0):
    """Lee los supuestos y calcula Cu, Co y q* por producto.

    Los factores permiten el análisis de sensibilidad: `cost_factor=1.2` sube el costo 20 %,
    `price_factor=0.8` baja el precio 20 % y `salvage_pct` es la fracción del precio que se
    recupera de lo que sobra (0 = se tira todo).
    """
    a = pd.read_csv(path)
    price = a["precio_gbp"] * price_factor
    cost = (a["precio_gbp"] * a["costo_pct"] * cost_factor).clip(upper=price)
    a = a.assign(precio=price, costo=cost)
    a["Cu"] = a["precio"] - a["costo"]
    a["Co"] = (a["costo"] - salvage_pct * a["precio"]).clip(lower=0)
    a["q_star"] = a["Cu"] / (a["Cu"] + a["Co"])
    return a.set_index("product_id")


def produce(prediction, rounding=np.ceil):
    """Unidades a producir: la predicción convertida a un entero con la regla indicada.

    La regla (hacia arriba, hacia abajo o al más cercano) se elige con la validación.
    """
    return rounding(np.clip(prediction, 0, None))


def quantile_choice(quantile_prediction, quantiles, q_star, interpolate=False):
    """Toma, para cada ejemplo, el escalón de la escalera que corresponde al q* de su producto.

    Sin interpolar se usa el escalón más cercano; interpolando, se combina linealmente con el
    siguiente escalón según qué tan lejos está q* de cada uno.
    """
    q = np.asarray(quantiles)
    q_star = np.clip(np.asarray(q_star), q[0], q[-1])
    if not interpolate:
        index = np.abs(q[None, :] - q_star[:, None]).argmin(axis=1)
        return quantile_prediction[np.arange(len(index)), ..., index]
    upper = np.clip(np.searchsorted(q, q_star), 1, len(q) - 1)
    lower = upper - 1
    weight = (q_star - q[lower]) / (q[upper] - q[lower])
    rows = np.arange(len(q_star))
    low, high = quantile_prediction[rows, ..., lower], quantile_prediction[rows, ..., upper]
    return low + weight[:, None] * (high - low)


def newsvendor_cost(produced, actual, cu, co):
    """Costo de cada producto-día: faltantes × Cu + sobrantes × Co."""
    shortage = np.clip(actual - produced, 0, None)
    leftover = np.clip(produced - actual, 0, None)
    return shortage * cu + leftover * co, shortage, leftover
