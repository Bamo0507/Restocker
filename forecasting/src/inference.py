"""Pronóstico para un negocio de Restocker con la iteración 5 (LSTM de 28 días con 9 cuantiles).

Arma la entrada igual que en el entrenamiento (windows.py y notebook 04):
- un día está abierto si el negocio vendió algo ese día; los días sin ventas cuentan como cerrados
  (los faltantes recientes ya los bloqueó forecast_readiness antes de llegar aquí),
- rasgos por día: venta normalizada, abierto, feriado y feriado en los próximos 2 días,
- escala = 1 + promedio de las ventas de los días abiertos de la ventana.
Los días futuros se asumen abiertos salvo cierre semanal o fecha marcada como cerrada.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import torch

from .models import LSTMForecaster
from .value import quantile_choice
from .windows import HORIZON, window_scale

LOOKBACK = 28
QUANTILES = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
MODEL_VERSION = "05_lstm_cuantiles"
CHECKPOINT = Path(__file__).resolve().parent.parent / "models" / "05_lstm_cuantiles.pt"


@dataclass
class Business:
    holiday_country: str
    holiday_subdivision: str | None = None
    closed_weekdays: list[int] = field(default_factory=list)  # ISO: 1 = lunes … 7 = domingo
    closed_dates: set[date] = field(default_factory=set)


def load_model(checkpoint=CHECKPOINT):
    """La red de la iteración 5 con la configuración que ganó la búsqueda (2 capas × 32)."""
    model = LSTMForecaster(past_features=4, future_features=3, quantiles=len(QUANTILES), hidden=32, layers=2)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.eval()


def holiday_flags(days, business):
    """Feriado y feriado en los próximos 2 días, con la misma regla del notebook 04."""
    years = range(days[0].year, days[-1].year + 2)
    calendar = holidays.country_holidays(business.holiday_country, subdiv=business.holiday_subdivision, years=years)
    dates = set(calendar.keys())
    is_holiday = np.array([d in dates for d in days], dtype=float)
    next_two = np.array([(d + timedelta(days=1)) in dates or (d + timedelta(days=2)) in dates for d in days], dtype=float)
    return is_holiday, next_two


def is_closed(day, business):
    return day.isoweekday() in business.closed_weekdays or day in business.closed_dates


def build_inputs(sales, product_ids, reference_date, business):
    """Entrada de la red para cada producto con al menos 28 días de historia.

    `sales` tiene product_id, sale_date y sold (la vista daily_sales).
    Devuelve los tensores, la escala, los productos usados y los que no tienen historia suficiente.
    """
    past_days = [reference_date - timedelta(days=LOOKBACK - 1 - i) for i in range(LOOKBACK)]
    future_days = [reference_date + timedelta(days=i + 1) for i in range(HORIZON)]

    sales = sales[sales["sold"] > 0]
    open_days = set(sales["sale_date"])
    past_open = np.array([d in open_days for d in past_days], dtype=float)
    future_open = np.array([not is_closed(d, business) for d in future_days], dtype=float)
    past_calendar = np.column_stack([past_open, *holiday_flags(past_days, business)])
    future_calendar = np.column_stack([future_open, *holiday_flags(future_days, business)])

    first_sale = sales.groupby("product_id")["sale_date"].min()
    table = sales.pivot_table(index="sale_date", columns="product_id", values="sold", aggfunc="sum")
    table = table.reindex(past_days).fillna(0.0)

    used, skipped, past, scale = [], [], [], []
    for product in product_ids:
        if product not in first_sale.index or first_sale[product] > past_days[0]:
            skipped.append(product)
            continue
        series = table[product].to_numpy(dtype=float) * past_open  # un día cerrado vende 0
        s = window_scale(series, past_open)
        past.append(np.column_stack([series / s, past_calendar]))
        scale.append(s)
        used.append(product)

    return (
        np.asarray(past, dtype=np.float32),
        np.repeat(future_calendar[None], len(used), axis=0).astype(np.float32),
        np.asarray(scale, dtype=np.float32),
        used,
        skipped,
        future_days,
    )


def forecast(model, sales, products, reference_date, business):
    """Filas para la tabla forecasts (sin business_id) y los productos sin historia suficiente.

    `products` es una tabla con id, unit_price, unit_cost y made_in_advance.
    """
    products = products.set_index("id")
    past, future, scale, used, skipped, future_days = build_inputs(sales, list(products.index), reference_date, business)
    if not used:
        return [], skipped

    with torch.no_grad():
        ladder = model(torch.from_numpy(past), torch.from_numpy(future)).numpy() * scale[:, None, None]
    ladder = np.maximum.accumulate(np.clip(ladder, 0, None), axis=2)  # la escalera nunca baja
    ladder[:, future[0, :, 0] == 0, :] = 0.0                          # un día cerrado no vende

    info = products.loc[used]
    price, cost = info["unit_price"].astype(float), info["unit_cost"].astype(float)
    has_value = info["made_in_advance"].astype(bool) & (price > 0) & cost.notna()
    q_star = ((price - cost) / price).clip(0, 1).where(has_value)
    recommended = np.round(quantile_choice(ladder, QUANTILES, q_star.fillna(0.5).to_numpy()))

    rows = []
    for i, product in enumerate(used):
        for h, day in enumerate(future_days):
            rows.append({
                "product_id": product,
                "forecast_date": day.isoformat(),
                "horizon": h + 1,
                "median": round(float(ladder[i, h, 0]), 3),
                "quantiles": {str(q): round(float(ladder[i, h, k]), 3) for k, q in enumerate(QUANTILES)},
                "recommended_quantity": float(recommended[i, h]) if has_value.iloc[i] else None,
                "q_star": round(float(q_star.iloc[i]), 3) if has_value.iloc[i] else None,
                "model_version": MODEL_VERSION,
            })
    return rows, skipped
