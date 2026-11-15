// Llaves de TanStack Query que varias features invalidan entre sí.
export const queryKeys = {
  catalog: ["catalog", "list"] as const,
  inventory: ["inventory", "list"] as const,
  history: ["history", "list"] as const,
  members: ["members", "list"] as const,
  capture: ["capture", "context"] as const,
  forecast: ["forecast", "board"] as const,
  salesWindow: (end: string) => ["forecast", "sales-window", end] as const,
};
