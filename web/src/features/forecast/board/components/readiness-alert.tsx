"use client";

import Link from "next/link";
import { CalendarX2, TriangleAlert } from "lucide-react";
import { Button, buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { formatDayList, formatShortDay } from "../utils/format-forecast";

// Días sin ventas registradas: no se asume que fueron cero. Por cada día se registra lo vendido
// (Capturar) o se marca como cerrado.
export function ReadinessAlert({
  days,
  blocking,
  closingDate,
  onClose,
}: {
  days: string[];
  blocking: boolean;
  closingDate: string | null;
  onClose: (date: string) => void;
}) {
  const title = blocking
    ? `Faltan las ventas del ${formatDayList(days)}`
    : `No hay ventas registradas del ${formatDayList(days)}`;
  const detail = blocking
    ? "La red necesita los últimos 7 días completos. Registra lo que se vendió o marca el día como cerrado."
    : "El pronóstico saldría desde el último día registrado. Registra esos días para que empiece mañana.";

  return (
    <section
      role={blocking ? "alert" : "status"}
      className={cn(
        "flex flex-col gap-4 rounded-card p-4",
        blocking ? "bg-status-out/15" : "bg-status-low/15",
      )}
    >
      <div className="flex gap-3">
        <TriangleAlert
          className={cn("mt-0.5 size-5 shrink-0", blocking ? "text-status-out" : "text-status-low")}
          aria-hidden
        />
        <div className="flex flex-col gap-1">
          <p className="font-heading text-lg font-bold leading-tight">{title}</p>
          <p className="text-sm text-muted-foreground">{detail}</p>
        </div>
      </div>

      <ul className="flex flex-col gap-2">
        {days.map((day) => (
          <li key={day} className="flex flex-wrap items-center justify-between gap-2 rounded-2xl bg-card px-3 py-2">
            <span className="text-sm font-medium">{formatShortDay(day)}</span>
            <div className="flex gap-2">
              <Link href="/capturar" className={buttonVariants({ variant: "secondary", size: "sm" })}>
                Registrar ventas
              </Link>
              <Button
                variant="ghost"
                size="sm"
                disabled={closingDate === day}
                onClick={() => onClose(day)}
              >
                <CalendarX2 aria-hidden />
                Estuvo cerrado
              </Button>
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}
