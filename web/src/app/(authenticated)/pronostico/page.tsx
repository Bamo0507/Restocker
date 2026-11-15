import { ForecastBoard } from "@/features/forecast/board/components/forecast-board";

export default async function PronosticoPage({ searchParams }: { searchParams: Promise<{ vista?: string }> }) {
  const { vista } = await searchParams;
  return <ForecastBoard initialView={vista === "ventas" ? "ventas" : "pronostico"} />;
}
