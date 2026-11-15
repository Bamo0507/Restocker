"""Servicio de pronóstico de Restocker.

POST /forecast con el token de sesión del usuario (Authorization: Bearer <token>):
1. Las lecturas van con el token del usuario, así el RLS de Supabase las limita a su negocio.
2. Si faltan ventas de los últimos 7 días (forecast_readiness), responde 409 con esos días.
3. Pronostica los próximos 7 días desde el último día registrado y reemplaza los pronósticos
   del negocio. Solo esta escritura usa la llave de servidor.

Variables de entorno: SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY, SUPABASE_SECRET_KEY.
En local: forecasting/run-local.sh las toma de web/.env.
"""

import os
from datetime import date, timedelta

import httpx
import pandas as pd
from fastapi import FastAPI, Header, HTTPException

from src.inference import LOOKBACK, Business, forecast, load_model

# En local se pueden reusar las variables de web/.env (con prefijo NEXT_PUBLIC_).
SUPABASE_URL = (os.environ.get("SUPABASE_URL") or os.environ["NEXT_PUBLIC_SUPABASE_URL"]).rstrip("/")
PUBLISHABLE_KEY = os.environ.get("SUPABASE_PUBLISHABLE_KEY") or os.environ["NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY"]
SECRET_KEY = os.environ["SUPABASE_SECRET_KEY"]
PAGE = 1000  # máximo de filas que Supabase devuelve por consulta

app = FastAPI(title="Restocker forecasting")
model = load_model()  # se carga una vez por instancia


def user_client(token):
    headers = {"apikey": PUBLISHABLE_KEY, "Authorization": f"Bearer {token}"}
    return httpx.Client(base_url=f"{SUPABASE_URL}/rest/v1", headers=headers, timeout=30)


def fetch_all(client, path, params):
    """Lee todas las filas, de 1,000 en 1,000."""
    rows, start = [], 0
    while True:
        response = client.get(path, params=params, headers={"Range": f"{start}-{start + PAGE - 1}"})
        response.raise_for_status()
        batch = response.json()
        rows += batch
        if len(batch) < PAGE:
            return rows
        start += PAGE


def bearer(authorization):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Falta el token de sesión")
    return authorization.split(" ", 1)[1]


@app.get("/")
def health():
    return {"ok": True}


@app.post("/forecast")
def run_forecast(authorization: str | None = Header(default=None)):
    token = bearer(authorization)
    with user_client(token) as db:
        user = httpx.get(f"{SUPABASE_URL}/auth/v1/user", headers=db.headers, timeout=30)
        if user.status_code != 200:
            raise HTTPException(401, "Sesión inválida o vencida")

        businesses = db.get("/businesses", params={
            "select": "id,holiday_country,holiday_subdivision,closed_weekdays"}).json()
        if not businesses:
            raise HTTPException(403, "El usuario no pertenece a un negocio")
        business = businesses[0]

        readiness = db.post("/rpc/forecast_readiness", json={}).json()
        if not readiness.get("reference_date"):
            raise HTTPException(422, "El negocio todavía no tiene ventas registradas")
        if readiness["missing_recent"]:
            raise HTTPException(409, {"message": "Faltan ventas de días recientes",
                                      "missing_recent": readiness["missing_recent"]})
        reference = date.fromisoformat(readiness["reference_date"])
        window_start = reference - timedelta(days=LOOKBACK - 1)

        products = pd.DataFrame(fetch_all(db, "/products", {
            "select": "id,name,unit_price,unit_cost,made_in_advance", "is_active": "eq.true", "order": "id"}))
        sales = pd.DataFrame(fetch_all(db, "/daily_sales", {
            "select": "product_id,sale_date,sold", "sale_date": f"lte.{reference.isoformat()}",
            "order": "product_id,sale_date"}),
            columns=["product_id", "sale_date", "sold"])
        closed = fetch_all(db, "/closed_days", {
            "select": "closed_on", "closed_on": f"gte.{window_start.isoformat()}", "order": "closed_on"})

    if products.empty:
        raise HTTPException(422, "El negocio no tiene productos activos")
    sales["sale_date"] = pd.to_datetime(sales["sale_date"]).dt.date
    sales["sold"] = sales["sold"].astype(float)
    calendar = Business(
        holiday_country=business["holiday_country"],
        holiday_subdivision=business["holiday_subdivision"],
        closed_weekdays=business["closed_weekdays"] or [],
        closed_dates={date.fromisoformat(row["closed_on"]) for row in closed},
    )

    rows, skipped = forecast(model, sales, products.drop(columns="name"), reference, calendar)
    for row in rows:
        row["business_id"] = business["id"]

    admin = {"apikey": SECRET_KEY, "Content-Type": "application/json", "Prefer": "return=minimal"}
    base = f"{SUPABASE_URL}/rest/v1/forecasts"
    httpx.delete(base, params={"business_id": f"eq.{business['id']}"}, headers=admin, timeout=30).raise_for_status()
    if rows:
        httpx.post(base, json=rows, headers=admin, timeout=60).raise_for_status()

    names = products.set_index("id")["name"]
    return {
        "reference_date": reference.isoformat(),
        "first_forecast_date": (reference + timedelta(days=1)).isoformat(),
        "products_forecasted": len({row["product_id"] for row in rows}),
        "skipped": [names[p] for p in skipped],
        "missing_older": readiness["missing_older"],
        "pending": readiness["pending"],
    }
