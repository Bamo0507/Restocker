// Las fechas del pronóstico son días (YYYY-MM-DD), sin hora: se leen en UTC para que no se corran.
function toDate(day: string) {
  return new Date(`${day}T00:00:00Z`);
}

const weekdayShort = new Intl.DateTimeFormat("es-GT", { weekday: "short", timeZone: "UTC" });
const weekdayLong = new Intl.DateTimeFormat("es-GT", { weekday: "long", timeZone: "UTC" });
const dayMonth = new Intl.DateTimeFormat("es-GT", { day: "numeric", month: "long", timeZone: "UTC" });

const capitalize = (text: string) => text.charAt(0).toUpperCase() + text.slice(1);

// "Lun"
export const formatWeekdayShort = (day: string) => capitalize(weekdayShort.format(toDate(day)).replace(".", ""));
// "16"
export const formatDayNumber = (day: string) => String(toDate(day).getUTCDate());
// "lunes 16 de noviembre"
export const formatLongDay = (day: string) => `${weekdayLong.format(toDate(day))} ${dayMonth.format(toDate(day))}`;
// "13 de noviembre"
export const formatShortDay = (day: string) => dayMonth.format(toDate(day));

// "13, 14 y 15 de noviembre" para avisos de varios días.
export function formatDayList(days: string[]) {
  const labels = days.map(formatShortDay);
  if (labels.length <= 1) return labels[0] ?? "";
  return `${labels.slice(0, -1).join(", ")} y ${labels.at(-1)}`;
}

// Las ventas son unidades enteras; el pronóstico se muestra redondeado.
export const formatUnits = (value: number) => Math.round(value).toLocaleString("es-GT");

// De 0.75 a "3 de cada 4 días"; para el resto, "75 de cada 100 días".
export function formatCoverage(q: number) {
  const percent = Math.round(q * 100);
  if (percent % 10 === 0) return `${percent / 10} de cada 10 días`;
  if (percent % 25 === 0) return `${percent / 25} de cada 4 días`;
  return `${percent} de cada 100 días`;
}

// El escalón de la escalera que usa la recomendación: el más cercano a q*.
export function nearestQuantile(quantiles: Record<string, number>, qStar: number) {
  return Object.keys(quantiles)
    .map(Number)
    .reduce((best, q) => (Math.abs(q - qStar) < Math.abs(best - qStar) ? q : best));
}

// Montos sin símbolo de moneda: el negocio todavía no guarda en qué moneda vende.
export const formatMoney = (value: number) =>
  value.toLocaleString("es-GT", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

// Cuántas veces cuesta más quedarse corto que sobrar: "3 veces", "2.6 veces".
export function formatRatio(ratio: number) {
  const rounded = Math.round(ratio * 10) / 10;
  return `${Number.isInteger(rounded) ? rounded : rounded.toLocaleString("es-GT")} veces`;
}
