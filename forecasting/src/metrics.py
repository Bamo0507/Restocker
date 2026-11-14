"""Métricas de error. Se calculan en unidades reales y solo en los días con el local abierto.

Las predicciones llegan en la escala normalizada (como las da el modelo) y se multiplican
por la escala de cada ventana para volver a unidades.
"""

import numpy as np
import pandas as pd


def to_units(windows, prediction):
    """Devuelve (predicción, venta real, máscara) en unidades, aplanadas día por día."""
    scale = windows.scale[:, None]
    return (prediction * scale).ravel(), (windows.target * scale).ravel(), windows.mask.ravel().astype(bool)


def mae(prediction, actual):
    """Error absoluto promedio: por cuántas unidades se equivoca, en promedio, por día."""
    return float(np.mean(np.abs(prediction - actual)))


def rmse(prediction, actual):
    """Raíz del error cuadrático medio: como el MAE, pero castiga más los errores grandes."""
    return float(np.sqrt(np.mean((prediction - actual) ** 2)))


def wape(prediction, actual):
    """Error absoluto como porcentaje de lo vendido. Funciona aunque haya días con 0 ventas."""
    return float(np.sum(np.abs(prediction - actual)) / np.sum(actual))


def evaluate(windows, prediction, naive_prediction, by=("split", "dataset")):
    """Tabla de métricas por grupo (por defecto, por parte y dataset) y en total.

    Cada fila de la tabla cuenta predicciones: un mismo día aparece en hasta 7 ventanas, una por
    cada distancia a la que se predijo (de 1 a 7 días antes).
    `MAE relativo` es el MAE dividido entre el de "igual que la semana pasada" sobre los mismos
    días: menos de 1 significa que mejora a esa línea base.
    """
    pred, actual, mask = to_units(windows, prediction)
    naive, _, _ = to_units(windows, naive_prediction)
    groups = windows.meta[list(by)].loc[windows.meta.index.repeat(windows.mask.shape[1])].reset_index(drop=True)
    days = groups.assign(pred=pred, actual=actual, naive=naive)[mask]

    def summarize(frame):
        return pd.Series({
            "predicciones": len(frame),
            "MAE": mae(frame["pred"], frame["actual"]),
            "RMSE": rmse(frame["pred"], frame["actual"]),
            "WAPE": wape(frame["pred"], frame["actual"]),
            "MAE relativo": mae(frame["pred"], frame["actual"]) / mae(frame["naive"], frame["actual"]),
        })

    table = days.groupby(list(by)).apply(summarize, include_groups=False)
    table.loc[("Total",) + ("",) * (len(by) - 1), :] = summarize(days)
    table["predicciones"] = table["predicciones"].astype(int)
    return table
