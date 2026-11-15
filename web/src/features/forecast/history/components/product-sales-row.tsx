"use client";

import { ChevronDown } from "lucide-react";
import { ProductThumb } from "@/features/shared/components/product-thumb";
import { cn } from "@/lib/utils";
import type { SalesWindowDay, SalesWindowProduct } from "../../shared/types/forecast";
import { formatLongDay, formatUnits } from "../../board/utils/format-forecast";
import { SalesBarChart } from "./sales-bar-chart";

// Un producto con su total en la ventana. Al tocarlo se despliegan sus ventas día a día.
export function ProductSalesRow({
  entry,
  days,
  openDays,
  expanded,
  onToggle,
}: {
  entry: SalesWindowProduct;
  days: SalesWindowDay[];
  openDays: number;
  expanded: boolean;
  onToggle: () => void;
}) {
  const average = openDays ? entry.total / openDays : 0;
  const daysWithSales = entry.daily.filter((value) => value > 0).length;
  const best = entry.daily.reduce((bestIndex, value, index) => (value > entry.daily[bestIndex] ? index : bestIndex), 0);

  return (
    <li className="rounded-card bg-card">
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={expanded}
        className="flex w-full cursor-pointer items-center gap-3 px-4 py-3 text-left"
      >
        <ProductThumb name={entry.product.name} imageUrl={entry.product.imageUrl} className="size-10" />
        <div className="flex min-w-0 flex-1 flex-col">
          <span className="truncate font-medium">{entry.product.name}</span>
          <span className="text-sm text-muted-foreground">
            {entry.total > 0 ? `Unas ${formatUnits(average)} por día abierto` : "Sin ventas en estos días"}
          </span>
        </div>
        <div className="flex flex-col items-end">
          <span className="text-xs font-medium text-muted-foreground">Total</span>
          <span className="font-heading text-2xl font-extrabold tabular-nums leading-none">{formatUnits(entry.total)}</span>
        </div>
        <ChevronDown
          className={cn("size-4 shrink-0 text-muted-foreground transition-transform", expanded && "rotate-180")}
          aria-hidden
        />
      </button>

      {expanded && (
        <div className="flex flex-col gap-3 border-t px-4 py-3 text-sm">
          <SalesBarChart days={days.map((day, index) => ({ ...day, total: entry.daily[index] }))} className="h-48" />
          {entry.total > 0 && (
            <p className="text-muted-foreground">
              Vendió en {daysWithSales} de {openDays} días abiertos. Su mejor día fue el{" "}
              <span className="font-medium text-foreground">
                {formatLongDay(days[best].date)}, con {formatUnits(entry.daily[best])}
              </span>
              .
            </p>
          )}
        </div>
      )}
    </li>
  );
}
