-- Antes de pronosticar hay que saber si la historia está completa. Un día sin ventas puede ser
-- un día cerrado o un día que nadie registró; no se asume ninguno de los dos: el negocio marca
-- los días cerrados y los demás días sin ventas cuentan como faltantes.
-- Las fechas se toman en hora de Guatemala.

-------------------------------------------------------------
-- Ventas por día
-------------------------------------------------------------

-- Ventas = movimientos de uso (cantidad negativa). Las reversiones se guardan con el mismo tipo y
-- la cantidad opuesta, así que al sumar se cancelan solas.
create view public.daily_sales
with (security_invoker = true)
as
select
  business_id,
  product_id,
  (created_at at time zone 'America/Guatemala')::date as sale_date,
  -sum(quantity) as sold
from public.movements
where type = 'uso'
group by business_id, product_id, (created_at at time zone 'America/Guatemala')::date;

revoke all on public.daily_sales from anon;
grant select on public.daily_sales to authenticated;

-------------------------------------------------------------
-- Días cerrados
-------------------------------------------------------------

create table public.closed_days (
  business_id uuid not null references public.businesses (id) on delete cascade,
  closed_on date not null,
  created_by uuid references public.profiles (id) on delete set null default auth.uid(),
  created_at timestamptz not null default now(),
  primary key (business_id, closed_on)
);

alter table public.closed_days enable row level security;

create policy "Miembros ven los días cerrados" on public.closed_days
  for select to authenticated using (business_id = public.current_business_id());

create policy "Miembros marcan días cerrados" on public.closed_days
  for insert to authenticated with check (business_id = public.current_business_id());

create policy "Miembros desmarcan días cerrados" on public.closed_days
  for delete to authenticated using (business_id = public.current_business_id());

revoke update on public.closed_days from authenticated;

-------------------------------------------------------------
-- ¿Se puede pronosticar?
-------------------------------------------------------------

-- La red mira los últimos 28 días. Referencia = último día con ventas o marcado como cerrado.
-- Faltante = día de la ventana sin ventas y sin marca de cerrado (desde el primer día con
-- actividad, para no contar como faltantes los días antes de que el negocio existiera).
-- Los faltantes de los últimos 7 días bloquean el pronóstico; los más viejos solo avisan y se
-- tratan como cerrados, igual que en el entrenamiento. Aparte se avisa de los días entre la
-- referencia y ayer que nadie registró.
create function public.forecast_readiness()
returns jsonb
language sql
stable
set search_path = ''
as $$
  with activity as (
    select sale_date as day
    from public.daily_sales
    where business_id = public.current_business_id() and sold > 0
    union
    select closed_on
    from public.closed_days
    where business_id = public.current_business_id()
  ),
  bounds as (
    select max(day) as reference_date, greatest(min(day), max(day) - 27) as window_start
    from activity
  ),
  missing as (
    select d::date as day
    from bounds, generate_series(bounds.window_start, bounds.reference_date, interval '1 day') as d
    where d::date not in (select day from activity)
  )
  select jsonb_build_object(
    'reference_date', b.reference_date,
    'window_start', b.window_start,
    -- Días entre la referencia y ayer sin registrar: el pronóstico saldría desde una fecha vieja.
    'pending', coalesce(
      (select jsonb_agg(d::date order by d)
       from generate_series(b.reference_date + 1,
                            (now() at time zone 'America/Guatemala')::date - 1,
                            interval '1 day') as d), '[]'),
    'missing_recent', coalesce(
      (select jsonb_agg(day order by day) from missing where day > b.reference_date - 7), '[]'),
    'missing_older', coalesce(
      (select jsonb_agg(day order by day) from missing where day <= b.reference_date - 7), '[]')
  )
  from bounds b
$$;

grant execute on function public.forecast_readiness() to authenticated;

-------------------------------------------------------------
-- Reiniciar pronósticos
-------------------------------------------------------------

-- Los miembros pueden borrar los pronósticos de su negocio para volver a generarlos.
create policy "Miembros borran los pronósticos" on public.forecasts
  for delete to authenticated using (business_id = public.current_business_id());
