"use client";

import { ChartLine, Loader2, RotateCcw, RotateCw, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { PageHeader } from "@/features/shared/components/page-header";
import { SegmentedControl } from "@/features/shared/components/segmented-control";
import { SalesWindowView } from "../../history/components/sales-window-view";
import type { ForecastEntry } from "../../shared/types/forecast";
import { type ForecastView, useForecastBoard } from "../hooks/useForecastBoard";
import { formatLongDay } from "../utils/format-forecast";
import { DayTabs } from "./day-tabs";
import { ForecastRow } from "./forecast-row";
import { ReadinessAlert } from "./readiness-alert";

const VIEWS: { value: ForecastView; label: string }[] = [
  { value: "pronostico", label: "Próximos 7 días" },
  { value: "ventas", label: "Últimos 28 días" },
];

export function ForecastBoard({ initialView }: { initialView: ForecastView }) {
  const board = useForecastBoard(initialView);

  return (
    <div className="mx-auto flex w-full max-w-2xl flex-col gap-5">
      <PageHeader
        title="Pronóstico"
        actions={
          board.view === "pronostico" &&
          board.hasForecast && (
            <Button variant="secondary" size="touch" onClick={board.reset} disabled={board.isResetting}>
              <RotateCcw aria-hidden />
              Reiniciar
            </Button>
          )
        }
      />

      {board.hasHistory && (
        <SegmentedControl
          label="Vista del pronóstico"
          options={VIEWS}
          value={board.view}
          onChange={board.changeView}
          className="w-full sm:max-w-sm"
        />
      )}

      {board.view === "ventas" && board.windowEnd ? (
        <SalesWindowView end={board.windowEnd} forecastProductIds={board.forecastProductIds} />
      ) : (
        <ForecastContent board={board} />
      )}
    </div>
  );
}

function ForecastContent({ board }: { board: ReturnType<typeof useForecastBoard> }) {
  const readiness = board.readiness;

  return (
    <>
      {board.isLoading && <BoardSkeleton />}

      {board.error && (
        <div role="alert" className="flex flex-col items-center gap-3 rounded-card bg-card px-6 py-10 text-center">
          <p className="font-medium">{board.error.message}</p>
          <Button variant="secondary" size="touch" onClick={() => board.refetch()}>
            <RotateCw aria-hidden />
            Reintentar
          </Button>
        </div>
      )}

      {readiness && !board.hasHistory && (
        <EmptyState
          title="Aún no hay ventas"
          text="El pronóstico aprende de las ventas de los últimos 28 días. Registra lo que se vende cada día desde Capturar."
        />
      )}

      {readiness?.referenceDate && (
        <>
          <p className="-mt-2 text-sm text-muted-foreground">
            Última venta registrada: <span className="font-medium text-foreground">{formatLongDay(readiness.referenceDate)}</span>
          </p>

          {readiness.missingRecent.length > 0 && (
            <ReadinessAlert days={readiness.missingRecent} blocking closingDate={board.closingDate} onClose={board.closeDay} />
          )}
          {readiness.pending.length > 0 && (
            <ReadinessAlert days={readiness.pending} blocking={false} closingDate={board.closingDate} onClose={board.closeDay} />
          )}

          {!board.hasForecast && (
            <EmptyState
              title="Sin pronóstico vigente"
              text="Calcula cuánto se va a vender en los próximos 7 días y cuánto conviene producir."
              action={
                <Button size="touch" onClick={board.generate} disabled={board.isGenerating || board.isBlocked}>
                  {board.isGenerating ? <Loader2 className="animate-spin" aria-hidden /> : <Sparkles aria-hidden />}
                  {board.isGenerating ? "Calculando…" : "Generar pronóstico"}
                </Button>
              }
            />
          )}
        </>
      )}

      {board.hasForecast && (
        <>
          {board.isStale && (
            <section
              role="status"
              className="flex flex-wrap items-center justify-between gap-3 rounded-card bg-secondary p-4 text-secondary-foreground"
            >
              <div className="flex flex-col">
                <p className="font-heading text-lg font-bold leading-tight">Hay ventas nuevas</p>
                <p className="text-sm">Se registraron ventas después del último pronóstico.</p>
              </div>
              <Button size="touch" onClick={board.generate} disabled={board.isGenerating || board.isBlocked}>
                {board.isGenerating ? <Loader2 className="animate-spin" aria-hidden /> : <RotateCw aria-hidden />}
                {board.isGenerating ? "Calculando…" : "Actualizar pronóstico"}
              </Button>
            </section>
          )}

          <DayTabs days={board.days} activeDate={board.activeDay?.date ?? null} onSelect={board.selectDay} />

          {board.activeDay && (
            <section aria-label={formatLongDay(board.activeDay.date)} className="flex flex-col gap-2">
              <h2 className="px-1 text-sm font-medium text-muted-foreground first-letter:uppercase">
                {formatLongDay(board.activeDay.date)}
              </h2>
              {board.activeDay.isClosed ? (
                <p className="rounded-card bg-card px-4 py-6 text-center text-sm text-muted-foreground">
                  Ese día el negocio está cerrado.
                </p>
              ) : (
                <>
                  <EntryGroup title="Para producir" entries={board.toProduce} board={board} />
                  <EntryGroup title="Se preparan al momento" entries={board.onDemand} board={board} />
                </>
              )}
            </section>
          )}

          <p className="text-xs text-muted-foreground">
            No considera promociones ni eventos especiales.
            {board.generatedAt &&
              ` Generado el ${new Date(board.generatedAt).toLocaleString("es-GT", { dateStyle: "medium", timeStyle: "short" })} con el modelo ${board.modelVersion}.`}
          </p>
        </>
      )}
    </>
  );
}

function EntryGroup({
  title,
  entries,
  board,
}: {
  title: string;
  entries: ForecastEntry[];
  board: ReturnType<typeof useForecastBoard>;
}) {
  if (entries.length === 0) return null;
  return (
    <div className="flex flex-col gap-2">
      <h3 className="px-1 pt-1 font-heading text-lg font-bold">{title}</h3>
      <ul className="flex flex-col gap-2">
        {entries.map((entry) => (
          <ForecastRow
            key={entry.product.id}
            entry={entry}
            expanded={board.expandedId === entry.product.id}
            onToggle={() => board.toggleExpanded(entry.product.id)}
          />
        ))}
      </ul>
    </div>
  );
}

function EmptyState({ title, text, action }: { title: string; text: string; action?: React.ReactNode }) {
  return (
    <div className="flex flex-col items-center gap-4 rounded-card bg-card px-6 py-12 text-center">
      <span className="flex size-16 items-center justify-center rounded-full bg-secondary text-secondary-foreground">
        <ChartLine className="size-8" aria-hidden />
      </span>
      <div className="flex flex-col gap-1">
        <p className="font-heading text-2xl font-bold">{title}</p>
        <p className="max-w-sm text-sm text-muted-foreground">{text}</p>
      </div>
      {action}
    </div>
  );
}

function BoardSkeleton() {
  return (
    <div aria-label="Cargando pronóstico" className="flex flex-col gap-3">
      <div className="flex gap-2">
        {Array.from({ length: 5 }, (_, index) => (
          <Skeleton key={index} className="h-16 w-16 rounded-2xl" />
        ))}
      </div>
      {Array.from({ length: 4 }, (_, index) => (
        <div key={index} className="flex items-center gap-3 rounded-card bg-card px-4 py-3">
          <Skeleton className="size-12 rounded-2xl" />
          <div className="flex flex-1 flex-col gap-2">
            <Skeleton className="h-4 w-2/5" />
            <Skeleton className="h-3 w-1/3" />
          </div>
          <Skeleton className="h-8 w-10" />
        </div>
      ))}
    </div>
  );
}
