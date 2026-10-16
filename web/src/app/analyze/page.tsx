"use client";

import type { DragEvent } from "react";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useRef, useState } from "react";

import type { MultiECGResponse } from "~/types/ecg";

const MAX_FILES = 12;

type DetectionStatus = "detecting" | "detected" | "unknown";

function getFileKey(file: File) {
  return [file.name, file.size, file.lastModified].join("-");
}

function getErrorMessage(data: unknown): string | undefined {
  if (typeof data !== "object" || data === null) {
    return undefined;
  }

  if ("error" in data && typeof data.error === "string") {
    return data.error;
  }

  if ("detail" in data) {
    const detail = data.detail;

    if (typeof detail === "string") {
      return detail;
    }

    if (
      typeof detail === "object" &&
      detail !== null &&
      "message" in detail &&
      typeof detail.message === "string"
    ) {
      return detail.message;
    }
  }

  return undefined;
}

function splitDelimitedLine(line: string, delimiter: string): string[] {
  const values: string[] = [];

  let current = "";
  let quoted = false;

  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];

    if (character === '"') {
      if (quoted && line[index + 1] === '"') {
        current += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }

      continue;
    }

    if (character === delimiter && !quoted) {
      values.push(current.trim());
      current = "";
      continue;
    }

    current += character;
  }

  values.push(current.trim());

  return values;
}

function detectDelimiter(line: string): string {
  const delimiters = [",", "\t", ";", "|"];

  let bestDelimiter = ",";
  let highestCount = 0;

  for (const delimiter of delimiters) {
    const count = line.split(delimiter).length;

    if (count > highestCount) {
      highestCount = count;
      bestDelimiter = delimiter;
    }
  }

  return bestDelimiter;
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);

  const middle = Math.floor(sorted.length / 2);

  if (sorted.length % 2 === 0) {
    return (sorted[middle - 1]! + sorted[middle]!) / 2;
  }

  return sorted[middle]!;
}

function formatSamplingRate(rate: number): string {
  if (Number.isInteger(rate)) {
    return String(rate);
  }

  return rate.toFixed(2);
}

async function detectCsvSamplingRate(file: File): Promise<number | null> {
  try {
    /*
     * Only inspect the first 512 KB.
     * That is plenty for detecting metadata and timestamps
     * without reading a huge ECG file into browser memory.
     */
    const text = await file.slice(0, 512 * 1024).text();

    const lines = text
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);

    if (lines.length < 2) {
      return null;
    }

    // =====================================================
    // 1. LOOK FOR SAMPLING-RATE METADATA
    // =====================================================

    const metadata = lines.slice(0, 30).join("\n");

    /*
     * Supported examples:
     *
     * sampling_rate=300
     * sampling_rate: 300
     * sampling rate = 300
     * sample_rate,300
     * fs=300
     * fs: 300
     */
    const metadataMatch = metadata.match(
      /(?:sampling[\s_-]*rate|sample[\s_-]*rate|sampling[\s_-]*frequency|sample[\s_-]*frequency|fs)\s*[:=,]\s*(\d+(?:\.\d+)?)/i,
    );

    if (metadataMatch?.[1]) {
      const rate = Number(metadataMatch[1]);

      if (Number.isFinite(rate) && rate >= 1 && rate <= 10000) {
        return rate;
      }
    }

    // =====================================================
    // 2. FIND CSV HEADER
    // =====================================================

    const headerIndex = lines.findIndex((line) => {
      if (line.startsWith("#")) {
        return false;
      }

      return /[a-zA-Z]/.test(line);
    });

    if (headerIndex < 0 || headerIndex >= lines.length - 1) {
      return null;
    }

    const headerLine = lines[headerIndex]!;

    const delimiter = detectDelimiter(headerLine);

    const headers = splitDelimitedLine(headerLine, delimiter);

    const normalizedHeaders = headers.map((header) =>
      header
        .trim()
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "_")
        .replace(/^_|_$/g, ""),
    );

    const dataLines = lines.slice(headerIndex + 1, headerIndex + 501);

    // =====================================================
    // 3. LOOK FOR AN EXPLICIT SAMPLING-RATE COLUMN
    // =====================================================

    const rateColumnNames = new Set([
      "sampling_rate",
      "sample_rate",
      "samplerate",
      "sampling_frequency",
      "sample_frequency",
      "fs",
      "hz",
    ]);

    const rateColumnIndex = normalizedHeaders.findIndex((header) =>
      rateColumnNames.has(header),
    );

    if (rateColumnIndex >= 0) {
      for (const line of dataLines) {
        const cells = splitDelimitedLine(line, delimiter);

        const value = Number(cells[rateColumnIndex]);

        if (Number.isFinite(value) && value >= 1 && value <= 10000) {
          return value;
        }
      }
    }

    // =====================================================
    // 4. FIND TIME COLUMN
    // =====================================================

    const timeColumnNames = new Set([
      "time",
      "timestamp",
      "second",
      "seconds",
      "sec",
      "secs",
      "time_s",
      "time_sec",
      "time_secs",
      "time_second",
      "time_seconds",
      "time_ms",
      "timestamp_ms",
      "milliseconds",
      "millisecond",
      "time_us",
      "timestamp_us",
      "microseconds",
      "microsecond",
    ]);

    const timeColumnIndex = normalizedHeaders.findIndex((header) =>
      timeColumnNames.has(header),
    );

    if (timeColumnIndex < 0) {
      return null;
    }

    const times: number[] = [];

    for (const line of dataLines) {
      const cells = splitDelimitedLine(line, delimiter);

      const value = Number(cells[timeColumnIndex]);

      if (Number.isFinite(value)) {
        times.push(value);
      }
    }

    if (times.length < 3) {
      return null;
    }

    // =====================================================
    // 5. CALCULATE MEDIAN TIME DIFFERENCE
    // =====================================================

    const differences: number[] = [];

    for (let index = 1; index < times.length; index += 1) {
      const previous = times[index - 1];
      const current = times[index];

      if (previous === undefined || current === undefined) {
        continue;
      }

      const difference = current - previous;

      if (Number.isFinite(difference) && difference > 0) {
        differences.push(difference);
      }
    }

    if (differences.length < 2) {
      return null;
    }

    const medianDelta = median(differences);

    const timeHeader = normalizedHeaders[timeColumnIndex] ?? "";

    let secondsPerUnit = 1;

    if (
      timeHeader === "time_ms" ||
      timeHeader === "timestamp_ms" ||
      timeHeader === "milliseconds" ||
      timeHeader === "millisecond"
    ) {
      secondsPerUnit = 0.001;
    }

    if (
      timeHeader === "time_us" ||
      timeHeader === "timestamp_us" ||
      timeHeader === "microseconds" ||
      timeHeader === "microsecond"
    ) {
      secondsPerUnit = 0.000001;
    }

    const secondsBetweenSamples = medianDelta * secondsPerUnit;

    if (secondsBetweenSamples <= 0) {
      return null;
    }

    const rate = 1 / secondsBetweenSamples;

    /*
     * ECG sampling rates normally fall comfortably
     * inside this range.
     *
     * This is only a sanity check. We do not guess
     * or round to 200/250/300/etc.
     */
    if (!Number.isFinite(rate) || rate < 1 || rate > 10000) {
      return null;
    }

    return Math.round(rate * 100) / 100;
  } catch (error) {
    console.error("Sampling-rate detection failed:", error);

    return null;
  }
}

export default function AnalyzePage() {
  const router = useRouter();

  const inputRef = useRef<HTMLInputElement>(null);

  const [files, setFiles] = useState<File[]>([]);

  const [sampleRates, setSampleRates] = useState<Record<string, string>>({});

  const [detectionStatuses, setDetectionStatuses] = useState<
    Record<string, DetectionStatus>
  >({});

  const [symptoms, setSymptoms] = useState("");

  const [dragging, setDragging] = useState(false);

  const [loading, setLoading] = useState(false);

  const [error, setError] = useState("");

  function addFiles(incoming: File[]) {
    const supported = incoming.filter((file) => {
      const name = file.name.toLowerCase();

      return name.endsWith(".mat") || name.endsWith(".csv");
    });

    if (supported.length !== incoming.length) {
      setError("Only .mat and .csv ECG files are supported.");
    } else {
      setError("");
    }

    const existingKeys = new Set(files.map(getFileKey));

    const newFiles = supported.filter(
      (file) => !existingKeys.has(getFileKey(file)),
    );

    const allowedCount = Math.max(0, MAX_FILES - files.length);

    const filesToAdd = newFiles.slice(0, allowedCount);

    setFiles((current) => [...current, ...filesToAdd]);

    /*
     * Detect sampling rate automatically for CSV.
     *
     * MAT files do not show an input because the
     * backend already handles them separately.
     */
    for (const file of filesToAdd) {
      if (!file.name.toLowerCase().endsWith(".csv")) {
        continue;
      }

      const key = getFileKey(file);

      setDetectionStatuses((current) => ({
        ...current,
        [key]: "detecting",
      }));

      void detectCsvSamplingRate(file).then((rate) => {
        if (rate === null) {
          setDetectionStatuses((current) => ({
            ...current,
            [key]: "unknown",
          }));

          return;
        }

        setDetectionStatuses((current) => ({
          ...current,
          [key]: "detected",
        }));

        setSampleRates((current) => {
          /*
           * If the user already typed something,
           * do not overwrite it.
           */
          if (current[key]?.trim()) {
            return current;
          }

          return {
            ...current,

            [key]: formatSamplingRate(rate),
          };
        });
      });
    }
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();

    setDragging(false);

    const dropped = Array.from(event.dataTransfer.files);

    addFiles(dropped);
  }

  function removeFile(file: File) {
    const key = getFileKey(file);

    setFiles((current) => current.filter((item) => getFileKey(item) !== key));

    setSampleRates((current) => {
      const next = {
        ...current,
      };

      delete next[key];

      return next;
    });

    setDetectionStatuses((current) => {
      const next = {
        ...current,
      };

      delete next[key];

      return next;
    });
  }

  function clearAll() {
    setFiles([]);

    setSampleRates({});

    setDetectionStatuses({});

    setSymptoms("");

    setError("");

    if (inputRef.current) {
      inputRef.current.value = "";
    }
  }

  async function analyzeECG() {
    if (files.length === 0) {
      setError("Upload at least one ECG recording.");

      return;
    }

    /*
     * Don't start while a CSV rate is still
     * being inspected.
     */
    const stillDetecting = files.some((file) => {
      const key = getFileKey(file);

      return detectionStatuses[key] === "detecting";
    });

    if (stillDetecting) {
      setError(
        "Sampling rate detection is still running. Try again in a moment.",
      );

      return;
    }

    setLoading(true);

    setError("");

    try {
      const formData = new FormData();

      for (const file of files) {
        formData.append("files", file);
      }

      /*
       * Rates stay aligned with upload order.
       *
       * MAT = null
       *
       * CSV detected / entered:
       * number
       *
       * CSV unknown:
       * null
       *
       * If null is sent, the backend gets another
       * chance to inspect the CSV itself.
       */
      const rates = files.map((file) => {
        const isCSV = file.name.toLowerCase().endsWith(".csv");

        if (!isCSV) {
          return null;
        }

        const value = sampleRates[getFileKey(file)];

        if (!value) {
          return null;
        }

        const number = Number(value);

        return Number.isFinite(number) && number > 0 ? number : null;
      });

      formData.append("sample_rates", JSON.stringify(rates));

      if (symptoms.trim()) {
        formData.append("symptoms", symptoms.trim());
      }

      const apiUrl =
        process.env.NEXT_PUBLIC_ECG_API_URL ?? "http://127.0.0.1:8000";

      const response = await fetch(`${apiUrl}/predict`, {
        method: "POST",

        body: formData,
      });

      const data: unknown = await response.json();

      if (!response.ok) {
        throw new Error(getErrorMessage(data) ?? "ECG analysis failed.");
      }

      const result = data as MultiECGResponse;

      sessionStorage.setItem("ecgAnalysisResult", JSON.stringify(result));

      sessionStorage.setItem("ecgReportedSymptoms", symptoms);

      router.push("/results");
    } catch (error) {
      if (error instanceof Error) {
        setError(error.message);
      } else {
        setError("Something went wrong.");
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-slate-50">
      {/* HEADER */}

      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-5">
          <Link href="/" className="font-bold text-slate-900">
            ECG Analyzer
          </Link>

          <Link
            href="/"
            className="text-sm font-medium text-slate-500 transition hover:text-slate-900"
          >
            Home
          </Link>
        </div>
      </header>

      <div className="mx-auto max-w-4xl px-6 py-12">
        {/* TITLE */}

        <div className="text-center">
          <p className="text-xs font-semibold tracking-widest text-slate-400 uppercase">
            ECG Analysis
          </p>

          <h1 className="mt-3 text-4xl font-bold tracking-tight text-slate-900">
            Upload ECG Recordings
          </h1>

          <p className="mx-auto mt-4 max-w-2xl leading-7 text-slate-500">
            Upload one recording or several recordings to compare ECG results
            over time.
          </p>

          <p className="mt-2 text-sm text-slate-400">Up to {MAX_FILES} files</p>
        </div>

        {/* DROP ZONE */}

        <div
          onDragOver={(event) => {
            event.preventDefault();

            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          className={`mt-10 cursor-pointer rounded-3xl border-2 border-dashed px-8 py-14 text-center transition ${
            dragging
              ? "border-slate-900 bg-slate-100"
              : "border-slate-300 bg-white hover:border-slate-500"
          }`}
        >
          <input
            ref={inputRef}
            type="file"
            multiple
            accept=".mat,.csv,text/csv"
            className="hidden"
            onChange={(event) => {
              addFiles(Array.from(event.target.files ?? []));

              event.target.value = "";
            }}
          />

          <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-slate-100 text-2xl text-slate-700">
            ↑
          </div>

          <h2 className="mt-5 text-lg font-semibold text-slate-800">
            Drop ECG files here
          </h2>

          <p className="mt-2 text-sm text-slate-400">or click to browse</p>

          <div className="mt-5 flex justify-center gap-2">
            <span className="rounded-full bg-slate-100 px-3 py-1 text-xs text-slate-500">
              .MAT
            </span>

            <span className="rounded-full bg-slate-100 px-3 py-1 text-xs text-slate-500">
              .CSV
            </span>
          </div>
        </div>

        {/* SELECTED FILES */}

        {files.length > 0 && (
          <section className="mt-6 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <div className="flex items-center justify-between">
              <div>
                <p className="font-semibold text-slate-900">
                  Selected Recordings
                </p>

                <p className="mt-1 text-sm text-slate-400">
                  {files.length} {files.length === 1 ? "file" : "files"}
                </p>
              </div>

              <button
                type="button"
                onClick={clearAll}
                disabled={loading}
                className="text-sm font-medium text-slate-500 hover:text-slate-900"
              >
                Clear all
              </button>
            </div>

            <div className="mt-5 space-y-3">
              {files.map((file, index) => {
                const key = getFileKey(file);

                const isCSV = file.name.toLowerCase().endsWith(".csv");

                const detectionStatus = detectionStatuses[key];

                return (
                  <div
                    key={key}
                    className="rounded-xl border border-slate-200 p-4"
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="min-w-0">
                        <div className="flex items-center gap-3">
                          <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-slate-100 text-xs font-semibold text-slate-500">
                            {index + 1}
                          </span>

                          <div className="min-w-0">
                            <p className="truncate font-medium text-slate-800">
                              {file.name}
                            </p>

                            <p className="mt-0.5 text-xs text-slate-400">
                              {(file.size / 1024).toFixed(1)} KB
                            </p>
                          </div>
                        </div>
                      </div>

                      <button
                        type="button"
                        disabled={loading}
                        onClick={() => removeFile(file)}
                        className="shrink-0 text-sm font-medium text-red-500 hover:text-red-700"
                      >
                        Remove
                      </button>
                    </div>

                    {/* CSV SAMPLING RATE */}

                    {isCSV && (
                      <div className="mt-4 border-t border-slate-100 pt-4">
                        <div className="flex items-center justify-between gap-3">
                          <label className="text-xs font-medium tracking-wide text-slate-400 uppercase">
                            Sampling Rate
                          </label>

                          {detectionStatus === "detecting" && (
                            <span className="rounded-full bg-blue-50 px-2.5 py-1 text-xs font-medium text-blue-700">
                              Detecting...
                            </span>
                          )}

                          {detectionStatus === "detected" && (
                            <span className="rounded-full bg-green-50 px-2.5 py-1 text-xs font-medium text-green-700">
                              Auto-detected
                            </span>
                          )}

                          {detectionStatus === "unknown" && (
                            <span className="rounded-full bg-amber-50 px-2.5 py-1 text-xs font-medium text-amber-700">
                              Not detected
                            </span>
                          )}
                        </div>

                        <div className="mt-2 flex items-center gap-3">
                          <input
                            type="number"
                            min="1"
                            step="any"
                            placeholder={
                              detectionStatus === "detecting"
                                ? "Detecting..."
                                : "Example: 300"
                            }
                            value={sampleRates[key] ?? ""}
                            onChange={(event) =>
                              setSampleRates((current) => ({
                                ...current,

                                [key]: event.target.value,
                              }))
                            }
                            className="w-full rounded-xl border border-slate-200 px-4 py-3 text-sm outline-none focus:border-slate-500"
                          />

                          <span className="text-sm text-slate-500">Hz</span>
                        </div>

                        {detectionStatus === "detected" && (
                          <p className="mt-2 text-xs text-green-600">
                            Detected automatically from the ECG file. You can
                            override it if needed.
                          </p>
                        )}

                        {detectionStatus === "detecting" && (
                          <p className="mt-2 text-xs text-slate-400">
                            Checking file metadata and timestamps for the
                            original sampling rate.
                          </p>
                        )}

                        {detectionStatus === "unknown" && (
                          <div className="mt-3">
                            <p className="text-xs text-amber-600">
                              Sampling rate could not be detected automatically.
                              Enter it manually or choose a common rate:
                            </p>

                            <div className="mt-3 flex gap-2">
                              {[150, 200, 300].map((rate) => (
                                <button
                                  key={rate}
                                  type="button"
                                  onClick={() =>
                                    setSampleRates((current) => ({
                                      ...current,
                                      [key]: String(rate),
                                    }))
                                  }
                                  className={`rounded-lg border px-4 py-2 text-sm font-medium transition ${
                                    sampleRates[key] === String(rate)
                                      ? "border-slate-900 bg-slate-900 text-white"
                                      : "border-slate-200 bg-white text-slate-600 hover:border-slate-400"
                                  }`}
                                >
                                  {rate} Hz
                                </button>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </section>
        )}

        {/* SYMPTOMS */}

        {files.length > 0 && (
          <section className="mt-5 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <div className="flex items-center gap-2">
              <p className="font-semibold text-slate-900">Symptoms</p>

              <span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs text-slate-400">
                Optional
              </span>
            </div>

            <p className="mt-2 text-sm leading-6 text-slate-400">
              Only enter symptoms actually experienced. The AI will not assume
              symptoms that were not provided.
            </p>

            <textarea
              rows={4}
              value={symptoms}
              onChange={(event) => setSymptoms(event.target.value)}
              placeholder="Example: palpitations, dizziness, fatigue..."
              className="mt-4 w-full resize-none rounded-xl border border-slate-200 px-4 py-3 text-sm outline-none focus:border-slate-500"
            />
          </section>
        )}

        {/* ERROR */}

        {error && (
          <div className="mt-5 rounded-xl bg-red-50 p-4 text-sm text-red-700">
            {error}
          </div>
        )}

        {/* ANALYZE */}

        {files.length > 0 && (
          <button
            type="button"
            onClick={analyzeECG}
            disabled={loading}
            className="mt-6 w-full rounded-xl bg-slate-900 px-6 py-4 font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading
              ? `Analyzing ${files.length} recording${
                  files.length > 1 ? "s" : ""
                }...`
              : `Analyze ${files.length} ECG${files.length > 1 ? "s" : ""} →`}
          </button>
        )}

        {/* PROCESSING */}

        {loading && (
          <div className="mt-5 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <div className="flex items-center gap-3">
              <div className="h-3 w-3 animate-pulse rounded-full bg-slate-900" />

              <p className="font-semibold text-slate-800">
                Processing recordings
              </p>
            </div>

            <p className="mt-2 text-sm leading-6 text-slate-400">
              Each ECG is being preprocessed and analyzed by all three models. A
              combined AI summary will then review the results.
            </p>
          </div>
        )}
      </div>
    </main>
  );
}
