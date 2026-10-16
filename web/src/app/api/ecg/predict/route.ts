import { NextResponse } from "next/server";

export const runtime = "nodejs";

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

    const data = await response.json();

    if (!response.ok) {
      let message = "ECG analysis failed.";

      if (typeof data.detail === "string") {
        message = data.detail;
      }

      if (
        data.detail &&
        typeof data.detail === "object" &&
        typeof data.detail.message === "string"
      ) {
        message = data.detail.message;
      }

      return NextResponse.json(
        {
          error: message,

          detail: data.detail,
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
