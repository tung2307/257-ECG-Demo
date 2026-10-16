import { NextResponse } from "next/server";

export const runtime = "nodejs";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

export async function POST(request: Request) {
  try {
    const incoming = await request.formData();

    const files = incoming.getAll("files");

    if (files.length === 0) {
      return NextResponse.json(
        {
          error: "At least one ECG file is required.",
        },
        {
          status: 400,
        },
      );
    }

    const outgoing = new FormData();

    for (const item of files) {
      if (item instanceof File) {
        outgoing.append("files", item, item.name);
      }
    }

    const sampleRates = incoming.get("sample_rates");

    if (typeof sampleRates === "string" && sampleRates.trim()) {
      outgoing.append("sample_rates", sampleRates);
    }

    const symptoms = incoming.get("symptoms");

    if (typeof symptoms === "string" && symptoms.trim()) {
      outgoing.append("symptoms", symptoms);
    }

    const apiUrl = process.env.ECG_API_URL ?? "http://127.0.0.1:8000";

    const response = await fetch(`${apiUrl}/predict`, {
      method: "POST",
      body: outgoing,
      cache: "no-store",
    });

    const data: unknown = await response.json();

    if (!response.ok) {
      let message = "ECG analysis failed.";
      let detail: unknown;

      if (isRecord(data)) {
        detail = data.detail;
      }

      if (typeof detail === "string") {
        message = detail;
      } else if (isRecord(detail) && typeof detail.message === "string") {
        message = detail.message;
      }

      return NextResponse.json(
        {
          error: message,
          detail,
        },
        {
          status: response.status,
        },
      );
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error("ECG API bridge:", error);

    return NextResponse.json(
      {
        error: "Could not connect to ECG model server.",
      },
      {
        status: 500,
      },
    );
  }
}
