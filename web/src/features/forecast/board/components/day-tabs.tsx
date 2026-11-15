"use client";

import type { ForecastDay } from "../../shared/types/forecast";
import { cn } from "@/lib/utils";
import { formatDayNumber, formatWeekdayShort } from "../utils/format-forecast";

// Los 7 días pronosticados; el primero es "mañana" respecto al último día registrado.
export function DayTabs({
  days,
  activeDate,
  onSelect,
}: {
  days: ForecastDay[];
  activeDate: string | null;
  onSelect: (date: string) => void;
}) {
  return (
    <div role="tablist" aria-label="Día del pronóstico" className="-mx-1 flex gap-2 overflow-x-auto px-1 pb-1">
      {days.map((day, index) => {
        const active = day.date === activeDate;
        return (
          <button
            key={day.date}
            type="button"
            role="tab"
            aria-selected={active}
            onClick={() => onSelect(day.date)}
            className={cn(
              "flex min-w-16 shrink-0 cursor-pointer flex-col items-center gap-0.5 rounded-2xl px-3 py-2 transition-colors",
              active ? "bg-primary text-primary-foreground" : "bg-card text-foreground hover:bg-muted",
              day.isClosed && !active && "text-muted-foreground",
            )}
          >
            <span className="text-xs font-medium">{index === 0 ? "Mañana" : formatWeekdayShort(day.date)}</span>
            <span className="font-heading text-xl font-bold tabular-nums leading-none">{formatDayNumber(day.date)}</span>
            {day.isClosed && <span className="text-[0.65rem]">Cerrado</span>}
          </button>
        );
      })}
    </div>
  );
}
