"use client";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { useCurrentSession } from "@/features/shared/hooks/useCurrentSession";
import { queryKeys } from "@/features/shared/query-keys";
import { getSalesWindow } from "../../forecast-api";

// Vista "Últimos 28 días": la ventana de 28 días detrás del pronóstico. `end` es su último día
// (la referencia del pronóstico vigente o, si no hay, la última venta registrada).
// `forecastProductIds` dice qué productos entraron al modelo; null si todavía no hay pronóstico.
export function useSalesWindow(end: string, forecastProductIds: Set<string> | null) {
  const { session } = useCurrentSession();
  const businessId = session?.business?.id;

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: queryKeys.salesWindow(end),
    queryFn: () => getSalesWindow(businessId!, end),
    enabled: Boolean(businessId),
  });

  // Producto desplegado con su gráfico; uno a la vez.
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const days = data?.days ?? [];

  const products = data?.products ?? [];
  const withSales = products.filter((entry) => entry.total > 0);
  // Productos activos sin ventas en la ventana o que el modelo dejó fuera por historia corta.
  const notUsed = forecastProductIds
    ? products.filter((entry) => !forecastProductIds.has(entry.product.id))
    : products.filter((entry) => entry.total === 0);
  const used = forecastProductIds ? withSales.filter((entry) => forecastProductIds.has(entry.product.id)) : withSales;

  const total = days.reduce((sum, day) => sum + day.total, 0);
  const openDays = data?.openDays ?? 0;

  return {
    isLoading,
    error,
    refetch,
    window: data ?? null,
    days,
    expandedId,
    toggleExpanded: (productId: string) => setExpandedId((current) => (current === productId ? null : productId)),
    total,
    openDays,
    closedDays: days.length - openDays,
    missingDays: days.filter((day) => day.status === "missing").length,
    average: openDays ? total / openDays : 0,
    used,
    notUsed,
  };
}
