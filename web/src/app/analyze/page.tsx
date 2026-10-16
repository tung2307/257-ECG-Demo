"use client";

import type { DragEvent } from "react";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useRef, useState } from "react";

import type { MultiECGResponse } from "~/types/ecg";

const MAX_FILES = 12;

function getFileKey(file: File) {
  return [file.name, file.size, file.lastModified].join("-");
}

function getErrorMessage(data: unknown): string | undefined {
  if (typeof data !== "object" || data === null) {
    return undefined;
  }

  if (!("error" in data)) {
    return undefined;
  }

  const error = data.error;

  return typeof error === "string" ? error : undefined;
}

export default function AnalyzePage() {
  const router = useRouter();

  const inputRef = useRef<HTMLInputElement>(null);

  const [files, setFiles] = useState<File[]>([]);

  const [sampleRates, setSampleRates] = useState<Record<string, string>>({});

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

    setFiles((current) => {
      const existing = new Set(current.map(getFileKey));

      const newFiles = supported.filter(
        (file) => !existing.has(getFileKey(file)),
      );

      return [...current, ...newFiles].slice(0, MAX_FILES);
    });
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
  }

  function clearAll() {
    setFiles([]);

    setSampleRates({});

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

    setLoading(true);

    setError("");

    try {
      const formData = new FormData();

      for (const file of files) {
        formData.append("files", file);
      }

      /*
       * Rates are aligned with
       * the file upload order.
       *
       * MAT = null
       * CSV blank = null
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

      const response = await fetch("/api/ecg/predict", {
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
          } `}
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

                    {/* CSV RATE */}
                    {isCSV && (
                      <div className="mt-4 border-t border-slate-100 pt-4">
                        <label className="text-xs font-medium tracking-wide text-slate-400 uppercase">
                          Sampling Rate
                        </label>

                        <div className="mt-2 flex items-center gap-3">
                          <input
                            type="number"
                            min="1"
                            placeholder="Example: 200"
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

                        <p className="mt-2 text-xs text-slate-400">
                          Leave blank if the CSV includes a numeric time column.
                        </p>
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
              Each ECG is being preprocessed and analyzed by both models. A
              combined AI summary will then review the results.
            </p>
          </div>
        )}
      </div>
    </main>
  );
}
