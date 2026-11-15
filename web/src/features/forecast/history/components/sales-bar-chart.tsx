"use client";

import { Bar, BarChart, CartesianGrid, ReferenceArea, XAxis, YAxis } from "recharts";
import { type ChartConfig, ChartContainer, ChartTooltip } from "@/components/ui/chart";
import { cn } from "@/lib/utils";
import type { WindowDayStatus } from "../../shared/types/forecast";
import { formatDayNumber, formatLongDay, formatUnits, formatWeekdayShort } from "../../board/utils/format-forecast";

export type ChartDay = { date: string; total: number; status: WindowDayStatus };

const STATUS_LABEL: Record<Exclude<WindowDayStatus, "open">, string> = {
  weekly: "Cierre semanal",
  closed: "Marcado como cerrado",
  missing: "Sin registro: se toma como día cerrado",
};

const config = { total: { label: "Unidades", color: "var(--primary)" } } satisfies ChartConfig;

// Unidades vendidas por día con eje Y y el número exacto al tocar una barra. Los días sin venta
// del negocio no se dibujan como barras en cero: se sombrea su columna (gris si cerró, ámbar si
// no se registró), para no confundirlos con días en que no se vendió nada.
export function SalesBarChart({ days, className }: { days: ChartDay[]; className?: string }) {
  // El eje X marca el inicio de cada semana de la ventana y el último día.
  const ticks = days.filter((_, index) => index % 7 === 0 || index === days.length - 1).map((day) => day.date);
  const hasClosed = days.some((day) => day.status === "weekly" || day.status === "closed");
  const hasMissing = days.some((day) => day.status === "missing");

  return (
    <div className="flex flex-col gap-2">
      <ChartContainer config={config} className={cn("aspect-auto h-56 w-full", className)}>
        <BarChart data={days} margin={{ top: 8, right: 4, bottom: 0, left: 0 }} accessibilityLayer>
          <CartesianGrid vertical={false} />
          {days
            .filter((day) => day.status !== "open")
            .map((day) => (
              <ReferenceArea
                key={day.date}
                x1={day.date}
                x2={day.date}
                fill={day.status === "missing" ? "var(--status-low)" : "var(--muted-foreground)"}
                fillOpacity={day.status === "missing" ? 0.35 : 0.12}
                ifOverflow="extendDomain"
              />
            ))}
          <XAxis
            dataKey="date"
            ticks={ticks}
            interval={0}
            tickLine={false}
            axisLine={false}
            tickMargin={8}
            tickFormatter={(date: string) => `${formatWeekdayShort(date)} ${formatDayNumber(date)}`}
          />
          <YAxis
            width={36}
            allowDecimals={false}
            tickLine={false}
            axisLine={false}
            tickFormatter={(value: number) => formatUnits(value)}
          />
          <ChartTooltip cursor={{ fill: "var(--muted)", fillOpacity: 0.6 }} content={<DayTooltip />} />
          <Bar dataKey="total" fill="var(--color-total)" radius={[3, 3, 0, 0]} maxBarSize={28} />
        </BarChart>
      </ChartContainer>

      {(hasClosed || hasMissing) && (
        <ul className="flex flex-wrap gap-x-4 gap-y-1 pl-9 text-xs text-muted-foreground">
          <LegendItem className="bg-primary" label="Unidades vendidas" />
          {hasClosed && <LegendItem className="bg-muted-foreground/25" label="Cerrado" />}
          {hasMissing && <LegendItem className="bg-status-low/60" label="Sin registro" />}
        </ul>
      )}
    </div>
  );
}

function DayTooltip({ active, payload }: { active?: boolean; payload?: readonly { payload?: ChartDay }[] }) {
  const day = payload?.[0]?.payload;
  if (!active || !day) return null;

  return (
    <div className="flex flex-col gap-0.5 rounded-lg border bg-background px-3 py-2 text-xs shadow-md">
      <span className="text-muted-foreground first-letter:uppercase">{formatLongDay(day.date)}</span>
      <span className="font-heading text-base font-bold leading-tight">
        {day.status === "open"
          ? `${formatUnits(day.total)} ${day.total === 1 ? "unidad" : "unidades"}`
          : STATUS_LABEL[day.status]}
      </span>
    </div>
  );
}

function LegendItem({ className, label }: { className: string; label: string }) {
  return (
    <li className="flex items-center gap-1.5">
      <span className={cn("size-2.5 rounded-sm", className)} aria-hidden />
      {label}
    </li>
  );
}
