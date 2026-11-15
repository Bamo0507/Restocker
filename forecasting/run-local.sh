#!/usr/bin/env bash
# Levanta el servicio de pronóstico en http://127.0.0.1:8000 con las variables de web/.env.
cd "$(dirname "$0")"
set -a
. ../web/.env
set +a
exec .venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
