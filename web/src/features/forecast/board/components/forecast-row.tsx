"use client";

import { ChevronDown } from "lucide-react";
import { ProductThumb } from "@/features/shared/components/product-thumb";
import { cn } from "@/lib/utils";
import type { ForecastEntry } from "../../shared/types/forecast";
import {
  formatCoverage,
  formatMoney,
  formatRange,
  formatRatio,
  formatUnits,
  nearestQuantile,
} from "../utils/format-forecast";

// Un producto en el día elegido. Con recomendación: "Haz N" y qué tantos días alcanza (lo que el
// modelo optimiza y lo que quedó calibrado). Sin ella (café, té): el día típico, que es la mediana.
// Al tocarlo se ve de dónde sale el número.
export function ForecastRow({
  entry,
  expanded,
  onToggle,
}: {
  entry: ForecastEntry;
  expanded: boolean;
  onToggle: () => void;
}) {
  const { product, median, quantiles, interval, recommended, qStar } = entry;
  const step = qStar !== null ? nearestQuantile(quantiles, qStar) : null;
  const compact = recommended === null;

  return (
    <li className="rounded-card bg-card">
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={expanded}
        className={cn("flex w-full cursor-pointer items-center gap-3 px-4 text-left", compact ? "py-2.5" : "py-3")}
      >
        <ProductThumb name={product.name} imageUrl={product.imageUrl} className={cn(compact && "size-10")} />
        <div className="flex min-w-0 flex-1 flex-col">
          <span className="truncate font-medium">{product.name}</span>
          {interval && <span className="text-sm tabular-nums">Se esperan {formatRange(interval)}</span>}
          <span className="text-sm text-muted-foreground">
            {step !== null ? `Alcanza ${formatCoverage(step)}` : `Día típico: unas ${formatUnits(median)}`}
          </span>
        </div>
        {recommended !== null && (
          <div className="flex flex-col items-end">
            <span className="text-xs font-medium text-muted-foreground">Haz</span>
            <span className="font-heading text-3xl font-extrabold tabular-nums leading-none text-primary">
              {formatUnits(recommended)}
            </span>
          </div>
        )}
        <ChevronDown
          className={cn("size-4 shrink-0 text-muted-foreground transition-transform", expanded && "rotate-180")}
          aria-hidden
        />
      </button>

      {expanded && (
        <div className="flex flex-col gap-3 border-t px-4 py-3 text-sm">
          {interval && (
            <p>
              <span className="font-medium">Se esperan vender {formatRange(interval)}.</span> 7 de cada 10 días
              la venta queda entre {formatUnits(interval.low)} y {formatUnits(interval.high)}.
            </p>
          )}
          <p>
            <span className="font-medium">Día típico: {formatUnits(median)}.</span> La mitad de los días se vende menos y
            la otra mitad más.
          </p>
          {recommended !== null && step !== null && <WhyThisAmount entry={entry} recommended={recommended} step={step} />}
          {!product.madeInAdvance && (
            <p className="text-muted-foreground">Se prepara al momento, así que no hace falta decidir cuánto producir.</p>
          )}
          {product.madeInAdvance && recommended === null && (
            <p className="text-muted-foreground">Agrega precio y costo en el catálogo para recibir una recomendación.</p>
          )}

          <Ladder quantiles={quantiles} highlight={step} />
        </div>
      )}
    </li>
  );
}

// El newsvendor con los números del producto: si falta uno se pierde el margen (precio − costo);
// si sobra uno se pierde lo que costó hacerlo.
function WhyThisAmount({ entry, recommended, step }: { entry: ForecastEntry; recommended: number; step: number }) {
  const { unitPrice: price, unitCost: cost } = entry.product;
  if (price === null || cost === null || cost <= 0) {
    return (
      <p>
        <span className="font-medium">Por qué {formatUnits(recommended)}:</span> quedarse corto cuesta más que sobrar,
        así que conviene cubrir {formatCoverage(step)} y no solo un día típico.
      </p>
    );
  }

  const shortage = price - cost;
  // Si cuesta más del 50 % del precio, sobrar pesa más; la escalera empieza en el día típico.
  if (shortage <= cost) {
    return (
      <p>
        <span className="font-medium">Por qué {formatUnits(recommended)}:</span> se vende a {formatMoney(price)} y
        producirlo cuesta {formatMoney(cost)}, así que sobrar cuesta tanto o más que quedarse corto. Se recomienda un día
        típico.
      </p>
    );
  }

  return (
    <p>
      <span className="font-medium">Por qué {formatUnits(recommended)}:</span> se vende a {formatMoney(price)} y
      producirlo cuesta {formatMoney(cost)}. Si falta uno, dejas de ganar {formatMoney(shortage)}; si sobra uno, pierdes{" "}
      {formatMoney(cost)}. Quedarse corto cuesta {formatRatio(shortage / cost)} más que sobrar, así que conviene cubrir{" "}
      {formatCoverage(step)} y no solo un día típico.
    </p>
  );
}

// La escalera de cuantiles como barras: cuánto alcanza para cubrir cada proporción de días.
function Ladder({ quantiles, highlight }: { quantiles: Record<string, number>; highlight: number | null }) {
  const steps = Object.entries(quantiles)
    .map(([q, value]) => ({ q: Number(q), value }))
    .sort((a, b) => a.q - b.q);
  const max = Math.max(...steps.map((s) => s.value), 1);

  return (
    <div className="flex flex-col gap-1">
      <p className="text-xs font-medium text-muted-foreground">Unidades para cubrir cada proporción de días</p>
      {steps.map(({ q, value }) => (
        <div key={q} className="flex items-center gap-2 text-xs">
          <span className="w-10 shrink-0 tabular-nums text-muted-foreground">{Math.round(q * 100)} %</span>
          <div className="h-2 flex-1 rounded-full bg-muted">
            <div
              className={cn("h-2 rounded-full", q === highlight ? "bg-primary" : "bg-primary/35")}
              style={{ width: `${(value / max) * 100}%` }}
            />
          </div>
          <span className={cn("w-8 shrink-0 text-right tabular-nums", q === highlight && "font-bold")}>
            {formatUnits(value)}
          </span>
        </div>
      ))}
    </div>
  );
}
