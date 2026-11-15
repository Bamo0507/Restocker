import { NextResponse } from "next/server";
import { createClient } from "@/lib/supabase/server";

function error(status: number, message: string, extra?: Record<string, unknown>) {
  return NextResponse.json({ error: message, ...extra }, { status });
}

// POST: genera el pronóstico del negocio del usuario. Reenvía su sesión al servicio de Python
// (forecasting/app.py), que lee con ese token (RLS) y escribe la tabla forecasts.
export async function POST() {
  const serviceUrl = process.env.FORECAST_SERVICE_URL;
  if (!serviceUrl) return error(500, "El servicio de pronóstico no está configurado.");

  const supabase = await createClient();
  const { data: auth } = await supabase.auth.getUser();
  if (!auth.user) return error(401, "Tu sesión expiró. Inicia sesión de nuevo.");
  const { data: session } = await supabase.auth.getSession();
  if (!session.session) return error(401, "Tu sesión expiró. Inicia sesión de nuevo.");

  let response: Response;
  try {
    response = await fetch(`${serviceUrl.replace(/\/$/, "")}/forecast`, {
      method: "POST",
      headers: { Authorization: `Bearer ${session.session.access_token}` },
      cache: "no-store",
    });
  } catch {
    return error(502, "El servicio de pronóstico no responde. Intenta de nuevo en un momento.");
  }

  const body = await response.json().catch(() => null);
  if (response.ok) return NextResponse.json(body);

  // 409: faltan ventas recientes; el detalle trae los días.
  const detail = body?.detail;
  if (response.status === 409 && detail?.missing_recent) {
    return error(409, "Faltan ventas de días recientes.", { missingRecent: detail.missing_recent });
  }
  const message = typeof detail === "string" ? detail : "No pudimos generar el pronóstico.";
  return error(response.status >= 500 ? 502 : response.status, message);
}
