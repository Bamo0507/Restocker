"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { usePathname, useRouter } from "next/navigation";
import { useMemo, useState } from "react";
import { toast } from "sonner";
import { useCurrentSession } from "@/features/shared/hooks/useCurrentSession";
import { queryKeys } from "@/features/shared/query-keys";
import { generateForecast, getForecastBoard, markDayClosed, resetForecast } from "../../forecast-api";

function previousDay(day: string) {
  const date = new Date(`${day}T00:00:00Z`);
  date.setUTCDate(date.getUTCDate() - 1);
  return date.toISOString().slice(0, 10);
}

// Pantalla de pronóstico: revisión de días faltantes, generar, reiniciar y el día seleccionado.
export type ForecastView = "pronostico" | "ventas";

export function useForecastBoard(initialView: ForecastView) {
  const queryClient = useQueryClient();
  const router = useRouter();
  const pathname = usePathname();
  const [view, setView] = useState<ForecastView>(initialView);
  const { session } = useCurrentSession();
  const businessId = session?.business?.id;

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: queryKeys.forecast,
    queryFn: getForecastBoard,
  });

  const [selectedDate, setSelectedDate] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const refresh = () => queryClient.invalidateQueries({ queryKey: queryKeys.forecast });

  const generate = useMutation({
    mutationFn: generateForecast,
    onSuccess: async (summary) => {
      setSelectedDate(null);
      await refresh();
      const skipped = summary.skipped.length ? ` · ${summary.skipped.length} sin historia suficiente` : "";
      toast.success(`Pronóstico listo para ${summary.products_forecasted} productos${skipped}`);
    },
    onError: async (err) => {
      await refresh(); // un 409 significa que la revisión de faltantes cambió
      toast.error(err.message);
    },
  });

  const reset = useMutation({
    mutationFn: () => resetForecast(businessId!),
    onSuccess: async () => {
      setSelectedDate(null);
      await refresh();
      toast.success("Pronóstico reiniciado");
    },
    onError: (err) => toast.error(err.message),
  });

  const closeDay = useMutation({
    mutationFn: (date: string) => markDayClosed(businessId!, date),
    onSuccess: async () => {
      await refresh();
      toast.success("Día marcado como cerrado");
    },
    onError: (err) => toast.error(err.message),
  });

  const days = data?.days ?? [];
  const readiness = data?.readiness;
  const activeDate = selectedDate ?? days[0]?.date ?? null;
  const activeDay = days.find((day) => day.date === activeDate) ?? null;
  // Arriba lo que hay que decidir (con recomendación, de mayor a menor); abajo lo que se prepara al momento.
  const toProduce = (activeDay?.entries ?? [])
    .filter((entry) => entry.recommended !== null)
    .sort((a, b) => b.recommended! - a.recommended!);
  const onDemand = (activeDay?.entries ?? []).filter((entry) => entry.recommended === null);

  // El pronóstico arranca el día después de su referencia. Si ya hay ventas registradas más
  // recientes, quedó viejo y conviene actualizarlo (con los mismos datos daría lo mismo).
  const forecastReference = days[0] ? previousDay(days[0].date) : null;
  const isStale = Boolean(forecastReference && readiness?.referenceDate && readiness.referenceDate > forecastReference);

  // La ventana de 28 días detrás del pronóstico vigente; sin pronóstico, la que usaría el próximo.
  const windowEnd = forecastReference ?? readiness?.referenceDate ?? null;
  const firstDay = data?.days[0];
  const forecastProductIds = useMemo(
    () => (firstDay ? new Set(firstDay.entries.map((entry) => entry.product.id)) : null),
    [firstDay],
  );

  return {
    view,
    changeView: (next: ForecastView) => {
      setView(next);
      router.replace(next === "ventas" ? `${pathname}?vista=ventas` : pathname, { scroll: false });
    },
    windowEnd,
    forecastProductIds,
    isLoading,
    error,
    refetch,
    readiness,
    // Faltan ventas recientes: no se puede generar hasta completarlas.
    isBlocked: Boolean(readiness?.missingRecent.length),
    hasHistory: Boolean(readiness?.referenceDate),
    days,
    hasForecast: days.length > 0,
    isStale,
    activeDay,
    toProduce,
    onDemand,
    selectDay: (date: string) => {
      setSelectedDate(date);
      setExpandedId(null);
    },
    generatedAt: data?.generatedAt ?? null,
    modelVersion: data?.modelVersion ?? null,
    expandedId,
    toggleExpanded: (productId: string) => setExpandedId((current) => (current === productId ? null : productId)),
    generate: () => generate.mutate(),
    isGenerating: generate.isPending,
    reset: () => reset.mutate(),
    isResetting: reset.isPending,
    closeDay: (date: string) => closeDay.mutate(date),
    closingDate: closeDay.isPending ? closeDay.variables : null,
  };
}
