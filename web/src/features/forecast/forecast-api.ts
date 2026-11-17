import { createClient } from "@/lib/supabase/client";
import { signProductImages } from "@/features/shared/utils/product-images";
import type {
  ForecastBoard,
  ForecastDay,
  ForecastEntry,
  ForecastReadiness,
  ForecastRunSummary,
  SalesWindow,
  SalesWindowDay,
} from "./shared/types/forecast";

type ReadinessRow = {
  reference_date: string | null;
  pending: string[];
  missing_recent: string[];
  missing_older: string[];
};

// GET: la revisión de días faltantes y el pronóstico vigente del negocio, agrupado por día.
export async function getForecastBoard(): Promise<ForecastBoard> {
  const supabase = createClient();

  const [readinessResult, forecastsResult] = await Promise.all([
    supabase.rpc("forecast_readiness"),
    supabase
      .from("forecasts")
      .select(
        "forecast_date, horizon, median, quantiles, interval_low, interval_high, recommended_quantity, q_star, model_version, generated_at, product:products(id, name, image_path, made_in_advance, unit_price, unit_cost)",
      )
      .order("forecast_date")
      .order("median", { ascending: false }),
  ]);

  if (readinessResult.error || forecastsResult.error) throw new Error("No pudimos cargar el pronóstico.");

  const raw = readinessResult.data as ReadinessRow;
  const readiness: ForecastReadiness = {
    referenceDate: raw.reference_date,
    pending: raw.pending ?? [],
    missingRecent: raw.missing_recent ?? [],
    missingOlder: raw.missing_older ?? [],
  };

  const rows = forecastsResult.data;
  const signed = await signProductImages(supabase, rows.map((row) => row.product.image_path));

  const byDate = new Map<string, ForecastEntry[]>();
  for (const row of rows) {
    const entry: ForecastEntry = {
      product: {
        id: row.product.id,
        name: row.product.name,
        imageUrl: row.product.image_path ? (signed.get(row.product.image_path) ?? null) : null,
        madeInAdvance: row.product.made_in_advance,
        unitPrice: row.product.unit_price === null ? null : Number(row.product.unit_price),
        unitCost: row.product.unit_cost === null ? null : Number(row.product.unit_cost),
      },
      date: row.forecast_date,
      horizon: row.horizon,
      median: Number(row.median),
      quantiles: row.quantiles as Record<string, number>,
      interval:
        row.interval_low === null || row.interval_high === null
          ? null
          : { low: Number(row.interval_low), high: Number(row.interval_high) },
      recommended: row.recommended_quantity === null ? null : Number(row.recommended_quantity),
      qStar: row.q_star === null ? null : Number(row.q_star),
    };
    byDate.set(entry.date, [...(byDate.get(entry.date) ?? []), entry]);
  }

  const days: ForecastDay[] = [...byDate].map(([date, entries]) => ({
    date,
    // El servicio deja en 0 todo lo de un día que se sabe cerrado.
    isClosed: entries.every((entry) => entry.median === 0 && Object.values(entry.quantiles).every((q) => q === 0)),
    entries,
  }));

  return {
    readiness,
    days,
    generatedAt: rows[0]?.generated_at ?? null,
    modelVersion: rows[0]?.model_version ?? null,
  };
}

// POST: pide al servicio de pronóstico (vía la ruta interna) los próximos 7 días.
export async function generateForecast(): Promise<ForecastRunSummary> {
  const response = await fetch("/api/forecasts", { method: "POST" });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error ?? "No pudimos generar el pronóstico.");
  return body as ForecastRunSummary;
}

// DELETE: borra el pronóstico vigente para poder generarlo de nuevo.
export async function resetForecast(businessId: string) {
  const { error } = await createClient().from("forecasts").delete().eq("business_id", businessId);
  if (error) throw new Error("No pudimos reiniciar el pronóstico.");
}

// Marca un día faltante como cerrado: deja de contar como faltante.
export async function markDayClosed(businessId: string, date: string) {
  const { error } = await createClient().from("closed_days").insert({ business_id: businessId, closed_on: date });
  if (error) throw new Error("No pudimos marcar el día como cerrado.");
}

export const WINDOW_DAYS = 28;

function addDays(day: string, amount: number) {
  const date = new Date(`${day}T00:00:00Z`);
  date.setUTCDate(date.getUTCDate() + amount);
  return date.toISOString().slice(0, 10);
}

// Lunes = 1 … domingo = 7, como closed_weekdays.
const isoWeekday = (day: string) => new Date(`${day}T00:00:00Z`).getUTCDay() || 7;

// GET: las ventas de los 28 días que terminan en `end`, por día y por producto activo.
export async function getSalesWindow(businessId: string, end: string): Promise<SalesWindow> {
  const supabase = createClient();
  const start = addDays(end, -(WINDOW_DAYS - 1));

  const [salesResult, productsResult, closedResult, businessResult] = await Promise.all([
    supabase.from("daily_sales").select("product_id, sale_date, sold").gte("sale_date", start).lte("sale_date", end),
    supabase.from("products").select("id, name, image_path").eq("is_active", true).order("name"),
    supabase.from("closed_days").select("closed_on").gte("closed_on", start).lte("closed_on", end),
    supabase.from("businesses").select("closed_weekdays").eq("id", businessId).single(),
  ]);

  if (salesResult.error || productsResult.error || closedResult.error || businessResult.error) {
    throw new Error("No pudimos cargar las ventas.");
  }

  const dates = Array.from({ length: WINDOW_DAYS }, (_, index) => addDays(start, index));
  const position = new Map(dates.map((date, index) => [date, index]));
  const byProduct = new Map<string, number[]>();
  for (const row of salesResult.data) {
    const index = row.sale_date ? position.get(row.sale_date) : undefined;
    if (!row.product_id || index === undefined || !row.sold || row.sold <= 0) continue;
    const daily = byProduct.get(row.product_id) ?? Array<number>(WINDOW_DAYS).fill(0);
    daily[index] += row.sold;
    byProduct.set(row.product_id, daily);
  }

  const markedClosed = new Set(closedResult.data.map((row) => row.closed_on));
  const weekly = new Set(businessResult.data.closed_weekdays);
  const days: SalesWindowDay[] = dates.map((date, index) => {
    const total = [...byProduct.values()].reduce((sum, daily) => sum + daily[index], 0);
    const status =
      total > 0 ? "open" : weekly.has(isoWeekday(date)) ? "weekly" : markedClosed.has(date) ? "closed" : "missing";
    return { date, total, status };
  });

  const signed = await signProductImages(supabase, productsResult.data.map((product) => product.image_path));
  const products = productsResult.data
    .map((product) => {
      const daily = byProduct.get(product.id) ?? Array<number>(WINDOW_DAYS).fill(0);
      return {
        product: {
          id: product.id,
          name: product.name,
          imageUrl: product.image_path ? (signed.get(product.image_path) ?? null) : null,
        },
        total: daily.reduce((sum, value) => sum + value, 0),
        daily,
      };
    })
    .sort((a, b) => b.total - a.total);

  return { start, end, days, products, openDays: days.filter((day) => day.status === "open").length };
}
