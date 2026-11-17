-------------------------------------------------------------
-- Intervalo de predicción del pronóstico
-------------------------------------------------------------

-- Rango de unidades que se espera vender 7 de cada 10 días, alrededor de la mediana (intervalo
-- conformal del notebook 07, sección 13). Describe la demanda; la recomendación sigue siendo el
-- escalón q*. Vacío en los pronósticos generados antes de esta migración.
alter table public.forecasts
  add column interval_low numeric(12, 3) check (interval_low >= 0),
  add column interval_high numeric(12, 3),
  add constraint forecasts_interval_order check (interval_low <= interval_high);
