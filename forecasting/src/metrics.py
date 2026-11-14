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
    días: menos de 1 significa que mejora a esa línea base. Suma todos los errores juntos, así
    que los productos que más venden pesan más.
    `MASE` hace la misma división producto por producto (por serie) y promedia, así que cada
    serie cuenta igual, venda 2 o 200 unidades al día. Como The Bread Basket no tiene periodo de
    entrenamiento, la escala de cada serie es el error de "igual que la semana pasada" en los
    mismos días evaluados, no en su historial.
    """
    pred, actual, mask = to_units(windows, prediction)
    naive, _, _ = to_units(windows, naive_prediction)
    columns = list(dict.fromkeys(list(by) + ["store_id", "product_id"]))
    groups = windows.meta[columns].loc[windows.meta.index.repeat(windows.mask.shape[1])].reset_index(drop=True)
    days = groups.assign(pred=pred, actual=actual, naive=naive)[mask]

    def mase(frame):
        errors = frame.assign(model_error=(frame["pred"] - frame["actual"]).abs(), naive_error=(frame["naive"] - frame["actual"]).abs())
        per_series = errors.groupby(["store_id", "product_id"])[["model_error", "naive_error"]].mean()
        ratio = per_series["model_error"] / per_series["naive_error"]
        return float(ratio.replace(np.inf, np.nan).mean())  # se omiten series donde la línea base no se equivocó nunca

    def summarize(frame):
        return pd.Series({
            "predicciones": len(frame),
            "MAE": mae(frame["pred"], frame["actual"]),
            "RMSE": rmse(frame["pred"], frame["actual"]),
            "WAPE": wape(frame["pred"], frame["actual"]),
            "MAE relativo": mae(frame["pred"], frame["actual"]) / mae(frame["naive"], frame["actual"]),
            "MASE": mase(frame),
        })

    rows = {key if len(by) > 1 else key[0]: summarize(frame) for key, frame in days.groupby(list(by))}
    table = pd.DataFrame(rows).T
    table.index.names = list(by)
    total = ("Total",) + ("",) * (len(by) - 1) if len(by) > 1 else "Total"
    table.loc[total, :] = summarize(days)
    table["predicciones"] = table["predicciones"].astype(int)
    return table
