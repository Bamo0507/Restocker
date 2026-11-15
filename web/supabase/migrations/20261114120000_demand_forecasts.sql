-- Pronóstico de demanda (proyecto final). Un proceso diario (servicio batch) calcula, con la red
-- entrenada en forecasting/, las ventas de los próximos 7 días de cada producto y las guarda aquí.
-- La app solo las lee.

-------------------------------------------------------------
-- Productos: precio, costo y si se producen por adelantado
-------------------------------------------------------------

-- Con precio y costo se calcula el cuantil que conviene producir: q* = (precio − costo) / precio.
alter table public.products
  add column unit_price numeric(12, 2) check (unit_price >= 0),
  add column unit_cost numeric(12, 2) check (unit_cost >= 0),
  -- Solo los productos que se producen antes de venderse (pan, pasteles) reciben recomendación de
  -- cuánto producir; los que se preparan al momento (café, té) solo muestran el pronóstico.
  add column made_in_advance boolean not null default true;

-------------------------------------------------------------
-- Negocios: calendario de feriados
-------------------------------------------------------------

-- Códigos de la librería holidays de Python (país ISO 3166 y, si aplica, región).
alter table public.businesses
  add column holiday_country text not null default 'GT' check (length(holiday_country) = 2),
  add column holiday_subdivision text;

-------------------------------------------------------------
-- Pronósticos
-------------------------------------------------------------

create table public.forecasts (
  id uuid primary key default gen_random_uuid(),
  business_id uuid not null references public.businesses (id) on delete cascade,
  product_id uuid not null,
  forecast_date date not null,
  -- A cuántos días del cálculo está: 1 = mañana … 7.
  horizon smallint not null check (horizon between 1 and 7),
  -- Venta estimada (cuantil 0.50): la mitad de los días se vende menos y la otra mitad más.
  median numeric(12, 3) not null check (median >= 0),
  -- Escalera completa: {"0.5": 2.1, "0.55": 2.4, …, "0.9": 4.9}.
  quantiles jsonb not null,
  -- Unidades a producir (escalón más cercano a q*, redondeado). Vacío si el producto no tiene precio
  -- y costo, o si no se produce por adelantado.
  recommended_quantity numeric(12, 3) check (recommended_quantity >= 0),
  q_star numeric(4, 3) check (q_star between 0 and 1),
  model_version text not null,
  generated_at timestamptz not null default now(),
  foreign key (product_id, business_id) references public.products (id, business_id) on delete cascade,
  -- Cada día el proceso reemplaza el pronóstico vigente de cada producto y fecha.
  unique (product_id, forecast_date)
);

create index forecasts_business_date_idx on public.forecasts (business_id, forecast_date);

-- Los miembros del negocio leen sus pronósticos. Nadie escribe desde la app: el proceso diario usa
-- la llave de servidor, que no pasa por RLS.
alter table public.forecasts enable row level security;

create policy "Miembros ven los pronósticos" on public.forecasts
  for select to authenticated using (business_id = public.current_business_id());
