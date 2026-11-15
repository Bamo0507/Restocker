"use client";

import { RotateCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { formatShortDay, formatUnits } from "../../board/utils/format-forecast";
import { useSalesWindow } from "../hooks/useSalesWindow";
import { ProductSalesRow } from "./product-sales-row";
import { SalesBarChart } from "./sales-bar-chart";

// "Últimos 28 días": los 28 días de historia sobre los que se calcula el pronóstico.
export function SalesWindowView({
  end,
  forecastProductIds,
}: {
  end: string;
  forecastProductIds: Set<string> | null;
}) {
  const view = useSalesWindow(end, forecastProductIds);

  if (view.isLoading) return <WindowSkeleton />;

  if (view.error || !view.window) {
    return (
      <div role="alert" className="flex flex-col items-center gap-3 rounded-card bg-card px-6 py-10 text-center">
        <p className="font-medium">{view.error?.message ?? "No pudimos cargar las ventas."}</p>
        <Button variant="secondary" size="touch" onClick={() => view.refetch()}>
          <RotateCw aria-hidden />
          Reintentar
        </Button>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-5">
      <p className="text-sm text-muted-foreground">
        {forecastProductIds ? "El pronóstico vigente se calculó" : "El próximo pronóstico se calculará"} con las ventas del{" "}
        <span className="font-medium text-foreground">
          {formatShortDay(view.window.start)} al {formatShortDay(view.window.end)}
        </span>
        , qué días abrió el negocio y los feriados que vienen.
      </p>

      <dl className="grid grid-cols-3 gap-2">
        <Stat label="Unidades vendidas" value={formatUnits(view.total)} />
        <Stat label="Días abiertos" value={`${view.openDays} de ${view.days.length}`} />
        <Stat label="Por día abierto" value={formatUnits(view.average)} />
      </dl>

      <section className="flex flex-col gap-3 rounded-card bg-card p-4">
        <div className="flex flex-col">
          <h2 className="font-heading text-lg font-bold leading-tight">Ventas del negocio por día</h2>
          <p className="text-sm text-muted-foreground">Toca una barra para ver las unidades de ese día.</p>
        </div>
        <SalesBarChart days={view.days} />
      </section>

      {view.missingDays > 0 && (
        <p className="-mt-2 text-sm text-muted-foreground">
          {view.missingDays === 1 ? "Un día quedó" : `${view.missingDays} días quedaron`} sin registrar. No se toman como
          cero ventas, sino como días cerrados.
        </p>
      )}

      {view.used.length > 0 && (
        <section className="flex flex-col gap-2">
          <div className="flex flex-col px-1">
            <h2 className="font-heading text-lg font-bold">Por producto</h2>
            <p className="text-sm text-muted-foreground">Toca un producto para ver sus ventas día a día.</p>
          </div>
          <ul className="flex flex-col gap-2">
            {view.used.map((entry) => (
              <ProductSalesRow
                key={entry.product.id}
                entry={entry}
                days={view.days}
                openDays={view.openDays}
                expanded={view.expandedId === entry.product.id}
                onToggle={() => view.toggleExpanded(entry.product.id)}
              />
            ))}
          </ul>
        </section>
      )}

      {view.notUsed.length > 0 && (
        <section className="flex flex-col gap-2">
          <div className="flex flex-col px-1">
            <h2 className="font-heading text-lg font-bold">
              {forecastProductIds ? "No entraron al pronóstico" : "Sin ventas en estos días"}
            </h2>
            <p className="text-sm text-muted-foreground">
              Para pronosticar un producto se necesitan sus ventas desde el inicio de estos 28 días.
            </p>
          </div>
          <ul className="flex flex-col gap-2">
            {view.notUsed.map((entry) => (
              <ProductSalesRow
                key={entry.product.id}
                entry={entry}
                days={view.days}
                openDays={view.openDays}
                expanded={view.expandedId === entry.product.id}
                onToggle={() => view.toggleExpanded(entry.product.id)}
              />
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-0.5 rounded-card bg-card px-3 py-3">
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className="font-heading text-2xl font-bold tabular-nums leading-none">{value}</dd>
    </div>
  );
}

function WindowSkeleton() {
  return (
    <div aria-label="Cargando ventas" className="flex flex-col gap-3">
      <div className="grid grid-cols-3 gap-2">
        {Array.from({ length: 3 }, (_, index) => (
          <Skeleton key={index} className="h-16 rounded-card" />
        ))}
      </div>
      <Skeleton className="h-60 rounded-card" />
    </div>
  );
}
