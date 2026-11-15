"""Carga The Bread Basket en Restocker como negocio de demostración.

Uso (desde forecasting/):
    .venv/bin/python scripts/load_bread_basket.py           # simulación: muestra lo que haría
    .venv/bin/python scripts/load_bread_basket.py --apply   # escribe en Supabase

Lee NEXT_PUBLIC_SUPABASE_URL y SUPABASE_SECRET_KEY de web/.env (o del entorno). Necesita que el
negocio "The Bread Basket" ya exista (creado desde el onboarding) y que no tenga productos.

Qué carga:
- calendario: feriados de Escocia (GB-SCT), sin cierre semanal; los 3 días que cerró, en closed_days,
- los 20 productos, con precio y costo en los 15 que se hacen por adelantado y alias en español,
- la historia corrida 3,514 días (502 semanas, mismo día de la semana) hasta el domingo 15 de
  noviembre de 2026. El lunes 3 y martes 4 de abril de 2017 (16 y 17 de noviembre) se dejan fuera
  para ingresarlos y compararlos en vivo,
- por producto y día abierto: una entrada a las 6:00 y un uso a las 12:00 con lo vendido, para que
  el stock cierre cada día en 0 y no quede negativo.
"""

import argparse
import os
from datetime import timedelta
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUSINESS_NAME = "The Bread Basket"
SHIFT = timedelta(days=3514)
LAST_LOADED = pd.Timestamp("2017-04-02")  # domingo → 15 de noviembre de 2026
TIMEZONE = "-06:00"                         # hora de Guatemala

MADE_TO_ORDER = {"Coffee", "Tea", "Hot chocolate", "Toast", "Juice"}
ALIASES = {
    "Bread": ["pan"], "Farm House": ["pan de granja"], "Pastry": ["bollería", "pastelito"],
    "Medialuna": ["medialunas"], "Scandinavian": ["pan escandinavo"], "Cake": ["pastel"],
    "Brownie": ["brownies"], "Muffin": ["muffins", "magdalena"], "Scone": ["scones"],
    "Cookies": ["galletas"], "Alfajores": ["alfajor"], "Truffles": ["trufas"], "Fudge": [],
    "Sandwich": ["sándwich", "sandwiches"], "Soup": ["sopa"], "Coffee": ["café"], "Tea": ["té"],
    "Hot chocolate": ["chocolate caliente"], "Toast": ["tostadas"], "Juice": ["jugo"],
}


def read_env():
    env = dict(os.environ)
    path = ROOT.parent / "web" / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                env.setdefault(key.strip(), value.strip().strip('"'))
    return env["NEXT_PUBLIC_SUPABASE_URL"].rstrip("/"), env["SUPABASE_SECRET_KEY"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="escribir en Supabase")
    args = parser.parse_args()

    url, key = read_env()
    db = httpx.Client(base_url=f"{url}/rest/v1", timeout=60, headers={
        "apikey": key, "Content-Type": "application/json", "Prefer": "return=representation"})

    def get(path, **params):
        response = db.get(path, params=params)
        response.raise_for_status()
        return response.json()

    def insert(path, rows):
        out = []
        for start in range(0, len(rows), 1000):
            response = db.post(path, json=rows[start:start + 1000])
            response.raise_for_status()
            out += response.json()
        return out

    # 1. El negocio y su administrador
    businesses = get("/businesses", select="id,name", name=f"eq.{BUSINESS_NAME}")
    assert len(businesses) == 1, f"Se esperaba 1 negocio llamado {BUSINESS_NAME}, hay {len(businesses)}"
    business_id = businesses[0]["id"]
    admins = get("/profiles", select="id", business_id=f"eq.{business_id}", role="eq.admin")
    assert admins, "El negocio no tiene administrador"
    admin_id = admins[0]["id"]
    existing = get("/products", select="id", business_id=f"eq.{business_id}")
    assert not existing, f"El negocio ya tiene {len(existing)} productos; no se carga dos veces"

    # 2. Productos con precio y costo
    names = pd.read_csv(ROOT / "data" / "processed" / "bread_basket_products.csv")
    value = pd.read_csv(ROOT / "data" / "value_assumptions.csv").set_index("product_id")
    products = []
    for row in names.itertuples():
        priced = row.product_id in value.index
        products.append({
            "business_id": business_id, "name": row.name, "unit": "unidad",
            "unit_price": float(value.loc[row.product_id, "precio_gbp"]) if priced else None,
            "unit_cost": round(float(value.loc[row.product_id, "precio_gbp"] * value.loc[row.product_id, "costo_pct"]), 2) if priced else None,
            "made_in_advance": row.name not in MADE_TO_ORDER,
        })

    # 3. Historia corrida
    sales = pd.read_parquet(ROOT / "data" / "processed" / "bread_basket.parquet")
    sales = sales[sales["date"] <= LAST_LOADED]
    closed = sorted((sales.loc[~sales["is_open"].astype(bool), "date"] + SHIFT).dt.date.unique())
    sold = sales[sales["sales"] > 0]

    print(f"Negocio: {BUSINESS_NAME} ({business_id})")
    print(f"Productos: {len(products)} ({sum(p['made_in_advance'] for p in products)} por adelantado)")
    print(f"Historia: {(sales['date'].min() + SHIFT).date()} a {(LAST_LOADED + SHIFT).date()}, "
          f"{len(sold):,} producto-días con venta → {2 * len(sold):,} movimientos")
    print(f"Días cerrados: {[d.isoformat() for d in closed]}")
    if not args.apply:
        print("Simulación: no se escribió nada. Corre con --apply para cargar.")
        return

    db.patch("/businesses", params={"id": f"eq.{business_id}"}, json={
        "holiday_country": "GB", "holiday_subdivision": "SCT", "closed_weekdays": []}).raise_for_status()
    created = insert("/products", products)
    ids = {p["name"]: p["id"] for p in created}
    by_dataset_id = dict(zip(names["product_id"], names["name"]))
    insert("/product_aliases", [
        {"product_id": ids[name], "business_id": business_id, "alias": alias, "origin": "manual"}
        for name, aliases in ALIASES.items() for alias in aliases])
    insert("/closed_days", [
        {"business_id": business_id, "closed_on": d.isoformat(), "created_by": admin_id} for d in closed])

    movements = []
    for row in sold.sort_values("date").itertuples():
        day = (row.date + SHIFT).date().isoformat()
        common = {"business_id": business_id, "product_id": ids[by_dataset_id[row.product_id]],
                  "source": "manual", "performed_by": admin_id}
        movements.append({**common, "type": "entrada", "quantity": float(row.sales), "created_at": f"{day}T06:00:00{TIMEZONE}"})
        movements.append({**common, "type": "uso", "quantity": -float(row.sales), "created_at": f"{day}T12:00:00{TIMEZONE}"})
    db.headers["Prefer"] = "return=minimal"
    for start in range(0, len(movements), 1000):
        db.post("/movements", json=movements[start:start + 1000]).raise_for_status()
    print(f"Listo: {len(created)} productos y {len(movements):,} movimientos cargados.")


if __name__ == "__main__":
    main()
