// Respuesta de forecast_readiness(): qué tan completa está la historia de ventas.
export type ForecastReadiness = {
  referenceDate: string | null; // último día con ventas o marcado como cerrado (YYYY-MM-DD)
  pending: string[]; // días entre la referencia y ayer sin registrar
  missingRecent: string[]; // faltantes de los últimos 7 días: bloquean el pronóstico
  missingOlder: string[]; // faltantes más viejos: solo avisan
};

export type ForecastProduct = {
  id: string;
  name: string;
  imageUrl: string | null;
  madeInAdvance: boolean;
  unitPrice: number | null;
  unitCost: number | null;
};

// Un producto en un día pronosticado.
export type ForecastEntry = {
  product: ForecastProduct;
  date: string;
  horizon: number;
  median: number;
  // Escalera de cuantiles: { "0.5": 2.1, …, "0.9": 4.9 }.
  quantiles: Record<string, number>;
  recommended: number | null;
  qStar: number | null;
};

export type ForecastDay = {
  date: string;
  isClosed: boolean;
  entries: ForecastEntry[];
};

export type ForecastBoard = {
  readiness: ForecastReadiness;
  days: ForecastDay[];
  generatedAt: string | null;
  modelVersion: string | null;
};

// Lo que responde el servicio al generar.
export type ForecastRunSummary = {
  reference_date: string;
  first_forecast_date: string;
  products_forecasted: number;
  skipped: string[];
  missing_older: string[];
  pending: string[];
};

// Cómo quedó cada día de la ventana para el modelo. Un día abierto es uno con ventas; los demás
// entran como cerrados, pero se distingue por qué para que no parezcan días de cero ventas.
export type WindowDayStatus = "open" | "weekly" | "closed" | "missing";

export type SalesWindowDay = {
  date: string;
  total: number;
  status: WindowDayStatus;
};

export type SalesWindowProduct = {
  product: { id: string; name: string; imageUrl: string | null };
  total: number;
  daily: number[]; // 28 valores, en el orden de los días
};

// Los 28 días que mira el modelo para pronosticar la semana siguiente.
export type SalesWindow = {
  start: string;
  end: string;
  days: SalesWindowDay[];
  products: SalesWindowProduct[];
  openDays: number;
};
