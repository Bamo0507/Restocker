"""Fotos de Unsplash para los productos de The Bread Basket (demo).

Uso (desde forecasting/):
    .venv/bin/python scripts/product_images.py           # descarga y comprime en data/demo_images/
    .venv/bin/python scripts/product_images.py --apply   # además las sube a Supabase y actualiza image_path

Compresión igual que la app (product-images.ts): recorte cuadrado al centro, 800 px, WebP y calidad
bajando de 0.8 hasta quedar en ~150 KB. Ruta en el bucket: <negocio>/<producto>/<uuid>.webp.
Licencia de Unsplash: uso libre; los créditos quedan en data/demo_images/creditos.csv.
"""

import argparse
import csv
import io
import sys
import uuid
from pathlib import Path

import httpx
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_bread_basket import BUSINESS_NAME, ROOT, read_env  # noqa: E402

OUT = ROOT / "data" / "demo_images"
SIZE, TARGET_BYTES = 800, 150 * 1024

# producto: (foto en images.unsplash.com, autor, enlace en Unsplash)
PHOTOS = {
    "Bread": ("photo-1775591648306-9d1889c0b019", "Alan Aprilio", "FJyjcrEBsMo"),
    "Farm House": ("photo-1556471013-0001958d2f12", "Mae Mu", "BxsGyQwjwaw"),
    "Pastry": ("photo-1776397280314-275d0f295fd9", "Kris Tian", "z_-zMS8nZSU"),
    "Medialuna": ("photo-1691480162735-9b91238080f6", "personalgraphic.com", "VzUE5RtCuBA"),
    "Scandinavian": ("photo-1630182266697-92508c01e2d1", "renee baude", "iEtcPCvEMDI"),
    "Cake": ("photo-1464349095431-e9a21285b5f3", "Annie Spratt", "6SHd7Q-l1UQ"),
    "Brownie": ("photo-1583516867196-ea9a73ecdc13", "Chaman Raj", "0lmVsDyqpVI"),
    "Muffin": ("photo-1727245243406-8a0551900c13", "Erwan NONON", "KPcqNNsf3Qg"),
    "Scone": ("photo-1585670492648-f34b4044a718", "Julie Johnson", "OLoq19jOd9g"),
    "Cookies": ("photo-1532499016263-f2c3e89de9cd", "Rai Vidanes", "tVy3x2Vu_z4"),
    "Alfajores": ("photo-1552552492-9c335658343d", "Gal Cohen", "Gr8mawv4d74"),
    "Truffles": ("photo-1656821991459-ae723d90d648", "Scarlett Alt", "DjXW0dj1VPM"),
    "Fudge": ("photo-1588440506444-572fb1b1488e", "Zaron", "-wWa12jXU9M"),
    "Sandwich": ("photo-1603903631889-b5f3ba4d5b9b", "Farhad Ibrahimzade", "07RKITgmScQ"),
    "Soup": ("photo-1625944228691-997b74fa5866", "Farhad Ibrahimzade", "--ZnV294AaQ"),
    "Coffee": ("photo-1572442388796-11668a67e53d", "Anubhav Arora", "RFLDagtOsMM"),
    "Tea": ("photo-1615484477112-677decb29c57", "Mockup Graphics", "DJyffTDrXQI"),
    "Hot chocolate": ("photo-1590080876209-c70d0f0bf6a3", "American Heritage Chocolate", "AOoNx98hdm8"),
    "Toast": ("photo-1525351484163-7529414344d8", "Ben Kolde", "FFqNATH27EM"),
    "Juice": ("photo-1600271886742-f049cd451bba", "ABHISHEK HAJARE", "kkrXVKK-jhg"),
}


def file_name(product):
    return product.lower().replace(" ", "_") + ".webp"


def compress(raw):
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    image = image.crop((left, top, left + side, top + side)).resize((SIZE, SIZE), Image.LANCZOS)
    quality = 80
    while True:
        buffer = io.BytesIO()
        image.save(buffer, "WEBP", quality=quality)
        if buffer.tell() <= TARGET_BYTES or quality <= 40:
            return buffer.getvalue()
        quality -= 10


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "creditos.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["producto", "archivo", "autor", "foto"])
        for product, (photo, author, slug) in PHOTOS.items():
            path = OUT / file_name(product)
            if not path.exists():
                url = f"https://images.unsplash.com/{photo}?ixlib=rb-4.1.0&q=85&fm=jpg&w=1600"
                path.write_bytes(compress(httpx.get(url, timeout=60).raise_for_status().content))
            writer.writerow([product, path.name, author, f"https://unsplash.com/photos/{slug}"])
            print(f"{product:14} {path.stat().st_size / 1024:6.1f} KB")


def upload():
    url, key = read_env()
    rest = httpx.Client(base_url=f"{url}/rest/v1", headers={"apikey": key, "Content-Type": "application/json"}, timeout=60)
    storage = httpx.Client(base_url=f"{url}/storage/v1", headers={"apikey": key, "Authorization": f"Bearer {key}"}, timeout=60)
    business_id = rest.get("/businesses", params={"select": "id", "name": f"eq.{BUSINESS_NAME}"}).json()[0]["id"]
    products = rest.get("/products", params={"select": "id,name,image_path", "business_id": f"eq.{business_id}"}).json()
    for product in products:
        if product["name"] not in PHOTOS or product["image_path"]:
            continue
        path = f"{business_id}/{product['id']}/{uuid.uuid4()}.webp"
        data = (OUT / file_name(product["name"])).read_bytes()
        storage.post(f"/object/product-images/{path}", content=data,
                     headers={"Content-Type": "image/webp"}).raise_for_status()
        rest.patch("/products", params={"id": f"eq.{product['id']}"}, json={"image_path": path}).raise_for_status()
        print(f"Subida: {product['name']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="subir a Supabase")
    args = parser.parse_args()
    prepare()
    if args.apply:
        upload()
