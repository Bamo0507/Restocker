-- Calendario del negocio: días de la semana en que cierra siempre (por ejemplo, domingos).
-- Junto con closed_days (fechas puntuales) define qué días no cuentan como faltantes y qué días
-- futuros se pronostican como cerrados. Lo edita el administrador (política "Admin edita su negocio").

-- Días ISO: 1 = lunes … 7 = domingo.
alter table public.businesses
  add column closed_weekdays smallint[] not null default '{}'
    check (closed_weekdays <@ array[1, 2, 3, 4, 5, 6, 7]::smallint[]);

-- Igual que la versión anterior, pero los días de cierre semanal cuentan como cerrados (no son
-- faltantes ni pendientes) y los cierres futuros ya marcados no mueven la fecha de referencia.
create or replace function public.forecast_readiness()
returns jsonb
language sql
stable
set search_path = ''
as $$
  with business as (
    select closed_weekdays from public.businesses where id = public.current_business_id()
  ),
  activity as (
    select sale_date as day
    from public.daily_sales
    where business_id = public.current_business_id() and sold > 0
    union
    select closed_on
    from public.closed_days
    -- Las fechas futuras ya marcadas (por ejemplo, 25 de diciembre) no mueven la referencia.
    where business_id = public.current_business_id()
      and closed_on <= (now() at time zone 'America/Guatemala')::date
  ),
  bounds as (
    select max(day) as reference_date, greatest(min(day), max(day) - 27) as window_start
    from activity
  ),
  missing as (
    select d::date as day
    from bounds, business, generate_series(bounds.window_start, bounds.reference_date, interval '1 day') as d
    where d::date not in (select day from activity)
      and extract(isodow from d)::smallint <> all (business.closed_weekdays)
  )
  select jsonb_build_object(
    'reference_date', b.reference_date,
    'window_start', b.window_start,
    -- Días entre la referencia y ayer sin registrar: el pronóstico saldría desde una fecha vieja.
    'pending', coalesce(
      (select jsonb_agg(d::date order by d)
       from business, generate_series(b.reference_date + 1,
                                      (now() at time zone 'America/Guatemala')::date - 1,
                                      interval '1 day') as d
       where extract(isodow from d)::smallint <> all (business.closed_weekdays)
         and d::date not in (select day from activity)), '[]'),
    'missing_recent', coalesce(
      (select jsonb_agg(day order by day) from missing where day > b.reference_date - 7), '[]'),
    'missing_older', coalesce(
      (select jsonb_agg(day order by day) from missing where day <= b.reference_date - 7), '[]')
  )
  from bounds b
$$;
