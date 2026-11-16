# Restocker

Restocker registra los movimientos de inventario de un negocio a partir de fotos, audios y texto. Un modelo multimodal (Gemini) propone cada movimiento (producto, tipo y cantidad), una persona lo revisa y, al confirmar, se actualiza el inventario. El piloto se hizo con Sakura, una repostería en Guatemala.

Además, un LSTM entrenado con ventas de panaderías pronostica cuánto se va a vender de cada producto en los próximos 7 días y recomienda cuánto producir según su precio y su costo.

- **Aplicación:** https://restocker-three.vercel.app
- **API de pronóstico:** https://restocker-forecast.vercel.app

Proyecto 3 y proyecto final del curso CC3092 Deep Learning y Sistemas Inteligentes, Universidad del Valle de Guatemala. Todo lo del proyecto final (el pronóstico de demanda) está en la rama [`final`](https://github.com/Bamo0507/Restocker/tree/final); la rama `main` corresponde a la entrega del Proyecto 3.

## Estructura

| Carpeta | Qué contiene |
| :- | :- |
| [`web/`](web) | La aplicación (Next.js y Supabase) y la evaluación del prompt de captura. Todos los comandos `pnpm` de este README se corren desde aquí. |
| [`forecasting/`](forecasting) | Módulo de pronóstico de demanda del proyecto final: análisis exploratorio, preprocesamiento, entrenamiento, evaluación y el servicio que sirve el modelo. Ver [Pronóstico de demanda](#pronóstico-de-demanda). |
| [`docs/`](docs) | Marca, diagramas, video y presentación. |

## Contenido del repositorio

| Parte | Dónde está |
| :- | :- |
| Ingesta y preprocesamiento de entradas | [`compress-image.ts`](web/src/features/capture/new/utils/compress-image.ts) (fotos a 1600 px y JPEG), [`useAudioRecorder.ts`](web/src/features/capture/new/hooks/useAudioRecorder.ts) (audio de hasta 90 s) y la ruta [`POST /api/captures`](web/src/app/api/captures/route.ts) (valida, descarga los archivos y normaliza el audio) |
| Índice de recuperación | No aplica: el sistema no usa RAG y el catálogo completo va en cada solicitud |
| Prompt final (captura-v6) | [`capture-prompt.ts`](web/src/features/capture/shared/llm/capture-prompt.ts), con el esquema de respuesta en [`capture-response-schema.ts`](web/src/features/capture/shared/llm/capture-response-schema.ts) y la llamada a Gemini en [`capture-model.ts`](web/src/features/capture/shared/llm/capture-model.ts) |
| Versiones anteriores del prompt (v0 a v5) | [`eval/prompts/`](web/eval/prompts) |
| Aplicación | Next.js en [`web/src/`](web/src), con la base de datos, las funciones y los buckets en [`supabase/migrations/`](web/supabase/migrations) |
| Conjuntos de evaluación | [`eval/iteration-tests/`](web/eval/iteration-tests) (10 casos) y [`eval/final-tests/`](web/eval/final-tests) (50 casos), con el catálogo congelado en [`eval/catalogo.json`](web/eval/catalogo.json) |
| Scripts de evaluación | [`eval/run.ts`](web/eval/run.ts) y [`eval/versions.ts`](web/eval/versions.ts) |
| Resultados reportados | [`eval/results/`](web/eval/results) |

Cada caso de evaluación es una carpeta con `caso.json` (entradas y resultado esperado) y sus archivos de foto o audio.

## Requisitos

- Node.js 22.12 o superior y pnpm 10.
- Un proyecto de [Supabase](https://supabase.com) y la [CLI de Supabase](https://supabase.com/docs/guides/cli).
- Una llave de la [API de Gemini](https://aistudio.google.com/apikey).
- Python 3.12 y [uv](https://docs.astral.sh/uv/) para el módulo de pronóstico.
- macOS para correr la evaluación, porque el script usa `sips` para comprimir las fotos igual que la aplicación.

## Instalación

```bash
cd web
pnpm install
cp .env.example .env
```

Llena `web/.env` con tus valores. Las llaves nunca se suben al repositorio: `.env` está en `.gitignore` y solo se versiona `.env.example`.

| Variable | Para qué |
| :- | :- |
| `NEXT_PUBLIC_SUPABASE_URL` | URL del proyecto de Supabase |
| `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY` | Llave pública de Supabase |
| `SUPABASE_SECRET_KEY` | Llave secreta de Supabase, solo del lado del servidor |
| `GEMINI_API_KEY` | Llave de la API de Gemini |
| `GEMINI_CAPTURE_MODEL` | Opcional. Modelo principal, por defecto `gemini-3.8-flash` |
| `GEMINI_FALLBACK_MODEL` | Opcional. Modelo de respaldo, por defecto `gemini-3.5-flash-lite` |
| `FORECAST_SERVICE_URL` | URL del servicio de pronóstico. En local, `http://127.0.0.1:8000` |

Luego crea las tablas, funciones y buckets en tu proyecto de Supabase:

```bash
pnpm db:link
pnpm db:push
```

## Ejecución

```bash
pnpm dev
```

La aplicación queda en http://localhost:3000. Crea una cuenta, registra el negocio y sus productos en Catálogo, y desde Capturar envía fotos, un audio o un texto.

La pantalla Pronóstico necesita además el servicio de pronóstico corriendo en otra terminal (ver [Servicio de pronóstico](#servicio-de-pronóstico)):

```bash
cd forecasting
./run-local.sh
```

## Reproducir las métricas

La evaluación no necesita Supabase, solo `GEMINI_API_KEY` en `.env`. Las métricas del informe se obtuvieron con Gemini 3.5 Flash-Lite:

```bash
# Sección 2.5: las siete versiones del prompt sobre los 10 casos de iteración
pnpm eval --set iteration-tests --model gemini-3.5-flash-lite

# Sección 2.4: baseline (v0) contra la versión final (v6) sobre los 50 casos finales
pnpm eval --set final-tests --versions v0,v6 --model gemini-3.5-flash-lite
```

Cada corrida crea `web/eval/results/<set>/<fecha>-<modelo>/` con:

- `resumen.md`: estado correcto, categoría de rechazo, precisión, recall, formato inválido, errores técnicos, latencia p50 y p95 y costo total por versión, más la lista de fallas;
- `raw/`: la respuesta de cada caso con su latencia, tokens, costo y calificación.

El F1 se calcula a partir de la precisión y el recall, y los promedios de latencia y tokens de la Tabla 1 del informe salen de los archivos de `raw/`. Las corridas reportadas están en `web/eval/results/iteration-tests/20261030-2131-gemini-3.5-flash-lite/` y `web/eval/results/final-tests/20261030-2353-gemini-3.5-flash-lite/`.

Opciones del script: `--set`, `--versions` (por ejemplo `v5,v6`), `--model`, `--cases` (casos específicos) y `--concurrency` (3 por defecto). Como el modelo no es determinista, una nueva corrida puede variar en algún caso respecto a lo reportado.

## Pronóstico de demanda

El módulo de pronóstico vive en `forecasting/` y usa Python 3.12. Para crear el entorno:

```bash
cd forecasting
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt
```

Los datasets ya vienen en el repositorio, así que no hace falta descargarlos:

- `data/raw/`: los datasets originales, tal como se obtuvieron de su fuente.
- `data/processed/`: cada dataset limpio y llevado al formato común (una fila por local, producto y día).

Para regenerarlos, se corren los notebooks de `notebooks/` en orden con el kernel de `.venv`. Cada uno descarga su dataset si no está en `data/raw/`, lo analiza y guarda el resultado en `data/processed/`.

| Notebook | Dataset | Papel |
| :- | :- | :- |
| `01_ddop_bakery.ipynb` | Cadena de panaderías alemana (Huber et al., 2019) | Entrenamiento y prueba con locales que el modelo no vio |
| `02_french_bakery.ipynb` | Panadería francesa (Kaggle, *French Bakery Daily Sales*) | Entrenamiento |
| `03_bread_basket.ipynb` | Cafetería y panadería de Edimburgo (OpenML, *The Bread Basket*) | Prueba final y negocio de la demostración en Restocker |
| `04_unificacion.ipynb` | Los tres anteriores | Junta los datasets, agrega los feriados de cada país y marca la división en `unified.parquet` |
| `05_lineas_base.ipynb` | `unified.parquet` | Mide las líneas base (igual que la semana pasada y promedio de los mismos días) y guarda la iteración 0 en `results/` |
| `06_entrenamiento.ipynb` | `unified.parquet` | Entrena las iteraciones 1 a 5 (MLP, búsqueda de hiperparámetros del LSTM, ventana de 28 días, día de la semana, pérdida por cuantiles) y las compara en validación |
| `07_evaluacion.ipynb` | `unified.parquet` y `value_assumptions.csv` | Evalúa una sola vez en prueba (locales reservados de ddop, The Bread Basket y temporada alta), calcula el valor en dinero con el modelo newsvendor, la sensibilidad, la explicabilidad y el costo de inferencia |

### Qué hay en cada carpeta

| Carpeta | Qué contiene |
| :- | :- |
| `src/` | Código compartido: ventanas (`windows.py`), líneas base, métricas, modelos y entrenamiento, valor en dinero (`value.py`), y la inferencia del servicio (`inference.py` y `numpy_model.py`) |
| `models/` | El checkpoint de la mejor época de cada iteración (`.pt`) con su configuración (`.json`). El modelo de la app es `05_lstm_cuantiles`, también exportado a NumPy (`.npz`) |
| `results/` | Un CSV por tabla del informe: `00` a `05` salen de la validación (notebooks 05 y 06) y `07_*` de la prueba final. `07_resumen_iteraciones.csv` junta todas las iteraciones |
| `data/value_assumptions.csv` | Precios y costos supuestos de The Bread Basket para el valor en dinero |
| `scripts/` | Exportación y verificación del modelo en NumPy, y carga del negocio de la demostración |

### Reproducir las métricas del pronóstico

Con el entorno de arriba, se corren en orden `05_lineas_base.ipynb`, `06_entrenamiento.ipynb` y `07_evaluacion.ipynb`. El 06 vuelve a entrenar y sobrescribe los checkpoints de `models/`; para reproducir las métricas del informe sin entrenar de nuevo, basta con correr el 07, que carga los checkpoints guardados. Los resultados quedan en `results/`.

### Servicio de pronóstico

`app.py` es un servicio FastAPI con un solo endpoint, `POST /forecast`, que la aplicación llama desde la ruta [`POST /api/forecasts`](web/src/app/api/forecasts/route.ts) con el token de sesión del usuario:

1. Lee las ventas por día, los productos y el calendario del negocio con el token del usuario, así que el RLS de Supabase limita la lectura a su negocio.
2. Revisa con `forecast_readiness()` que estén registrados los últimos 7 días; si faltan, responde 409 con esos días y la app pide completarlos.
3. Arma las ventanas de 28 días igual que en el entrenamiento, pronostica los 7 días siguientes con la escalera de 9 cuantiles, elige el escalón de cada producto con el newsvendor (q\* = 1 − costo/precio) y reemplaza los pronósticos del negocio. Solo esta escritura usa la llave secreta.

El modelo se entrena con PyTorch, pero el servicio lo corre con NumPy: torch pesa unos 700 MB y Vercel acepta hasta 500 MB por función. `src/numpy_model.py` repite el forward del LSTM con los mismos pesos. Para regenerar los pesos y comprobar que da lo mismo que PyTorch:

```bash
.venv/bin/python scripts/export_weights.py
.venv/bin/python scripts/check_numpy_model.py
```

La verificación compara las dos versiones en las 146,547 ventanas de los tres datasets; la diferencia máxima es de 2.4e-6.

Para correr el servicio en local se agregan sus dependencias al mismo entorno y se levanta con las variables de `web/.env`:

```bash
uv pip install --python .venv fastapi==0.163.0 httpx==0.28.1 uvicorn
./run-local.sh
```

Queda en http://127.0.0.1:8000. Las dependencias que se instalan en Vercel están en `pyproject.toml` y `uv.lock`, sin torch.

### Negocio de la demostración

The Bread Basket se carga en Restocker como negocio de demostración, con sus 20 productos y su historia corrida al presente (misma secuencia de días de la semana). Primero se crea la cuenta y el negocio "The Bread Basket" desde la app, sin productos, y luego:

```bash
.venv/bin/python scripts/load_bread_basket.py --apply
.venv/bin/python scripts/product_images.py --apply
```

Sin `--apply` los dos scripts solo muestran lo que harían. El primero carga el calendario, los productos con su precio y costo, y las ventas como movimientos; el segundo sube las fotos de los productos (de Unsplash, con sus créditos en `data/demo_images/creditos.csv`).

## Despliegue

La aplicación y el servicio son dos proyectos de Vercel conectados al mismo repositorio, que se despliegan desde la rama `final`:

| Proyecto | Root Directory | Framework | Variables de entorno |
| :- | :- | :- | :- |
| `restocker` | `web` | Next.js | Las de `web/.env.example`, con `FORECAST_SERVICE_URL` apuntando a la URL de producción del servicio |
| `restocker-forecast` | `forecasting` | FastAPI | `SUPABASE_URL`, `SUPABASE_PUBLISHABLE_KEY` y `SUPABASE_SECRET_KEY` |

- `forecasting/.vercelignore` deja fuera los datos, notebooks, resultados y checkpoints; el servicio solo sube `app.py`, `src/` y los pesos `05_lstm_cuantiles.npz`.
- La aplicación llama al servicio desde el servidor, así que el navegador nunca ve su URL y no hace falta CORS. Debe ser la URL de producción, porque las de preview tienen protección de acceso.
- En Supabase, Authentication → URL Configuration lleva la URL de la aplicación como Site URL y en Redirect URLs, para que los enlaces de los correos apunten al sitio desplegado.

## Scripts de desarrollo

| Script | Qué hace |
| :- | :- |
| `pnpm dev` | Servidor de desarrollo |
| `pnpm build` y `pnpm start` | Build y servidor de producción |
| `pnpm validate` | Lint y revisión de tipos |
| `pnpm eval` | Evaluación del prompt |
| `pnpm db:link`, `pnpm db:push` y `pnpm db:types` | Conectar Supabase, aplicar las migraciones y regenerar los tipos |
