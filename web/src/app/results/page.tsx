"use client";

import { useEffect, useState } from "react";

import { useRouter } from "next/navigation";

import type {
  ECGRecordingResult,
  HistoryAISummary,
  ModelResult,
  MultiECGResponse,
} from "~/types/ecg";

type Tab = "overview" | "models" | "ai";

const modelMetrics: Record<
  string,
  {
    accuracy: number;
    macroF1: number;
    paperF1?: number;
  }
> = {
  "Random Forest": {
    accuracy: 0.5955,
    macroF1: 0.2505,
  },

  "CNN + Transformer": {
    accuracy: 0.8535,
    macroF1: 0.7963,
    paperF1: 0.823,
  },

  CTRhythm: {
    accuracy: 0.8658,
    macroF1: 0.7709,
    paperF1: 0.8427,
  },
};

export default function ResultsPage() {
  const router = useRouter();

  const [data, setData] = useState<MultiECGResponse | null>(null);

  const [selectedIndex, setSelectedIndex] = useState(0);

  const [activeTab, setActiveTab] = useState<Tab>("overview");

  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    const stored = sessionStorage.getItem("ecgAnalysisResult");

    if (!stored) {
      setLoaded(true);
      return;
    }

    try {
      const parsed = JSON.parse(stored) as MultiECGResponse;

      setData(parsed);
    } catch {
      sessionStorage.removeItem("ecgAnalysisResult");
    }

    setLoaded(true);
  }, []);

  if (!loaded) {
    return (
      <main className="flex min-h-screen items-center justify-center bg-slate-50">
        <p className="text-sm text-slate-400">Loading results...</p>
      </main>
    );
  }

  if (!data || data.recordings.length === 0) {
    return (
      <main className="flex min-h-screen items-center justify-center bg-slate-50 px-6">
        <div className="text-center">
          <p className="text-xs font-semibold tracking-widest text-slate-400 uppercase">
            ECG Analyzer
          </p>

          <h1 className="mt-3 text-3xl font-bold text-slate-900">
            No ECG Results
          </h1>

          <p className="mt-3 text-slate-500">Analyze ECG recordings first.</p>

          <button
            onClick={() => router.push("/analyze")}

            className="mt-6 rounded-xl bg-slate-900 px-6 py-3 font-semibold text-white transition hover:bg-slate-800"
          >
            Analyze ECG
          </button>
        </div>
      </main>
    );
  }

  const safeIndex = Math.min(selectedIndex, data.recordings.length - 1);

  const selected = data.recordings[safeIndex]!;

  return (
    <main className="min-h-screen bg-slate-50">
      {/* =================================================
          HEADER
      ================================================= */}
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
          <div>
            <a href="/" className="font-bold text-slate-900">
              ECG Analyzer
            </a>

            <p className="text-xs text-slate-400">CMPE 257 Research Project</p>
          </div>

          <button
            onClick={() => {
              sessionStorage.removeItem("ecgAnalysisResult");

              router.push("/analyze");
            }}

            className="rounded-xl border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-600 transition hover:bg-slate-50"
          >
            New Analysis
          </button>
        </div>
      </header>

      <div className="mx-auto max-w-7xl px-6 py-7">
        {/* =================================================
            PAGE TOP
        ================================================= */}
        <section className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="rounded-full bg-emerald-50 px-2.5 py-1 text-xs font-medium text-emerald-700">
                Analysis Complete
              </span>

              <span className="text-xs text-slate-400">
                {data.recording_count}{" "}
                {data.recording_count === 1 ? "recording" : "recordings"}
              </span>
            </div>

            <h1 className="mt-2 text-3xl font-bold tracking-tight text-slate-900">
              ECG Results
            </h1>

            <p className="mt-1 text-sm text-slate-500">
              Compare all three models and review the AI-assisted summary.
            </p>
          </div>

          {/* IMPORTANT */}
          <div className="max-w-xl rounded-xl border border-amber-100 bg-amber-50 px-4 py-3">
            <div className="flex items-start gap-3">
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-white text-xs font-bold text-amber-700">
                !
              </div>

              <div>
                <p className="text-sm font-semibold text-slate-800">
                  Research use only
                </p>

                <p className="mt-0.5 text-xs leading-5 text-slate-500">
                  These results are generated by research ECG models and are not
                  a medical diagnosis.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* =================================================
            RECORDING SELECTOR
        ================================================= */}
        <section className="mt-7">
          <div className="mb-3 flex items-center justify-between">
            <div>
              <p className="text-xs font-semibold tracking-wide text-slate-400 uppercase">
                Recordings
              </p>

              <p className="mt-0.5 text-sm text-slate-500">
                Select an ECG to inspect
              </p>
            </div>

            <span className="text-xs text-slate-400">Upload order</span>
          </div>

          <div className="flex gap-3 overflow-x-auto pb-2">
            {data.recordings.map((recording, index) => (
              <RecordingTab
                key={`${recording.sequence}-${recording.filename}`}

                recording={recording}

                active={index === safeIndex}

                onClick={() => {
                  setSelectedIndex(index);

                  setActiveTab("overview");
                }}
              />
            ))}
          </div>
        </section>

        {/* =================================================
            SELECTED RECORDING HEADER
        ================================================= */}
        <section className="mt-6 rounded-2xl border border-slate-200 bg-white px-5 py-4 shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="min-w-0">
              <p className="text-xs tracking-wide text-slate-400 uppercase">
                Recording {selected.sequence}
              </p>

              <h2 className="mt-1 truncate text-lg font-semibold text-slate-900">
                {selected.filename}
              </h2>
            </div>

            <div className="flex flex-wrap gap-5">
              <MiniMetric
                label="Duration"
                value={`${selected.signal.duration_seconds}s`}
              />

              <MiniMetric
                label="Input"
                value={`${selected.signal.original_sample_rate} Hz`}
              />

              <MiniMetric
                label="Model"
                value={`${selected.signal.sample_rate} Hz`}
              />

              <MiniMetric
                label="Samples"
                value={selected.signal.processed_samples.toLocaleString()}
              />
            </div>
          </div>
        </section>

        {/* =================================================
            TABS
        ================================================= */}
        <section className="mt-6">
          <div className="inline-flex rounded-xl border border-slate-200 bg-white p-1">
            <TabButton
              active={activeTab === "overview"}

              onClick={() => setActiveTab("overview")}
            >
              Overview
            </TabButton>

            <TabButton
              active={activeTab === "models"}

              onClick={() => setActiveTab("models")}
            >
              Models
            </TabButton>

            <TabButton
              active={activeTab === "ai"}

              onClick={() => setActiveTab("ai")}
            >
              AI Review
            </TabButton>
          </div>
        </section>

        {/* =================================================
            OVERVIEW
        ================================================= */}
        {activeTab === "overview" && (
          <OverviewTab
            selected={selected}

            summary={data.ai_summary}

            recordingCount={data.recording_count}

            onViewModels={() => setActiveTab("models")}

            onViewAI={() => setActiveTab("ai")}
          />
        )}

        {/* =================================================
            MODELS
        ================================================= */}
        {activeTab === "models" && <ModelsTab recording={selected} />}

        {/* =================================================
            AI REVIEW
        ================================================= */}
        {activeTab === "ai" && (
          <AIReviewTab
            summary={data.ai_summary}

            aiError={data.ai_error}

            recordingCount={data.recording_count}
          />
        )}

        {/* =================================================
            FILE ERRORS
        ================================================= */}
        {data.failed_count > 0 && (
          <details className="mt-6 rounded-xl border border-slate-200 bg-white px-5 py-4">
            <summary className="cursor-pointer text-sm font-semibold text-amber-700">
              {data.failed_count} recording
              {data.failed_count > 1 ? "s" : ""} could not be analyzed
            </summary>

            <div className="mt-3 space-y-2">
              {data.file_errors.map((item) => (
                <p
                  key={`${item.sequence}-${item.filename}`}
                  className="text-sm text-slate-500"
                >
                  {item.filename}: {item.error}
                </p>
              ))}
            </div>
          </details>
        )}

        <footer className="mt-8 border-t border-slate-200 py-5 text-center">
          <p className="text-xs text-slate-400">
            Educational and research use only. Not intended for medical
            diagnosis.
          </p>
        </footer>
      </div>
    </main>
  );
}

/* =========================================================
   OVERVIEW
========================================================= */

function OverviewTab({
  selected,
  summary,
  recordingCount,
  onViewModels,
  onViewAI,
}: {
  selected: ECGRecordingResult;

  summary: HistoryAISummary | null;

  recordingCount: number;

  onViewModels: () => void;

  onViewAI: () => void;
}) {
  return (
    <div className="mt-5 grid gap-5 lg:grid-cols-[1.5fr_1fr]">
      {/* MODEL SNAPSHOT */}
      <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex items-center justify-between gap-4">
          <div>
            <p className="text-xs font-semibold tracking-wide text-slate-400 uppercase">
              Model Snapshot
            </p>

            <h2 className="mt-1 text-lg font-semibold text-slate-900">
              Rhythm Classification
            </h2>
          </div>

          <button
            onClick={onViewModels}

            className="text-sm font-medium text-slate-500 transition hover:text-slate-900"
          >
            Details →
          </button>
        </div>

        <div className="mt-5 grid gap-3 md:grid-cols-3">
          {selected.models.map((model) => (
            <ModelSnapshot
              key={model.name}

              model={model}
            />
          ))}
        </div>
      </section>

      {/* AI SNAPSHOT */}
      <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex items-center justify-between gap-4">
          <div>
            <p className="text-xs font-semibold tracking-wide text-slate-400 uppercase">
              AI Review
            </p>

            <h2 className="mt-1 text-lg font-semibold text-slate-900">
              {recordingCount > 1 ? "History Summary" : "Result Summary"}
            </h2>
          </div>

          {summary && (
            <button
              onClick={onViewAI}

              className="text-sm font-medium text-slate-500 transition hover:text-slate-900"
            >
              Details →
            </button>
          )}
        </div>

        {summary ? (
          <>
            <p className="mt-4 line-clamp-4 text-sm leading-6 text-slate-600">
              {summary.overview}
            </p>

            <div className="mt-5 flex items-center justify-between rounded-xl bg-slate-50 px-4 py-3">
              <span className="text-sm text-slate-500">
                Suggested attention
              </span>

              <UrgencyBadge urgency={summary.urgency} />
            </div>
          </>
        ) : (
          <p className="mt-4 text-sm text-slate-400">AI summary unavailable.</p>
        )}
      </section>
    </div>
  );
}

/* =========================================================
   MODELS
========================================================= */

function ModelsTab({ recording }: { recording: ECGRecordingResult }) {
  return (
    <section className="mt-5">
      <div className="grid gap-5 lg:grid-cols-3">
        {recording.models.map((model) => (
          <ModelCard
            key={model.name}

            result={model}
          />
        ))}
      </div>
    </section>
  );
}

/* =========================================================
   AI REVIEW
========================================================= */

function AIReviewTab({
  summary,
  aiError,
  recordingCount,
}: {
  summary: HistoryAISummary | null;

  aiError: string | null;

  recordingCount: number;
}) {
  if (!summary) {
    return (
      <section className="mt-5 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <p className="font-semibold text-slate-900">AI review unavailable</p>

        <p className="mt-2 text-sm leading-6 text-slate-500">
          {aiError ??
            "The ECG models completed successfully, but no AI summary was generated."}
        </p>
      </section>
    );
  }

  return (
    <section className="mt-5">
      {/* SUMMARY */}
      <div className="grid gap-5 lg:grid-cols-[2fr_1fr]">
        <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <p className="text-xs font-semibold tracking-wide text-slate-400 uppercase">
            {recordingCount > 1 ? "Combined Review" : "Summary"}
          </p>

          <p className="mt-3 leading-7 text-slate-700">{summary.overview}</p>

          <div className="mt-5 grid gap-5 border-t border-slate-100 pt-5 md:grid-cols-2">
            <CompactText title="Trend" text={summary.trend} />

            <CompactText
              title="Model Agreement"
              text={summary.model_agreement}
            />
          </div>
        </div>

        {/* ATTENTION */}
        <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <p className="text-xs font-semibold tracking-wide text-slate-400 uppercase">
            Suggested Attention
          </p>

          <div className="mt-4">
            <UrgencyBadge urgency={summary.urgency} />
          </div>

          <p className="mt-4 text-sm leading-6 text-slate-500">
            This is educational AI guidance based on research model outputs. It
            is not a clinical risk score.
          </p>
        </div>
      </div>

      {/* CONDITIONS */}
      {summary.possible_conditions.length > 0 && (
        <div className="mt-5">
          <div>
            <h3 className="font-semibold text-slate-900">
              Possible Conditions to Discuss
            </h3>

            <p className="mt-1 text-xs text-slate-400">
              Rhythm-related possibilities only. These are not diagnoses.
            </p>
          </div>

          <div className="mt-3 grid gap-3 md:grid-cols-2 lg:grid-cols-3">
            {summary.possible_conditions.map((condition) => (
              <ConditionCard
                key={condition.name}

                name={condition.name}

                relevance={condition.relevance}

                reason={condition.reason}
              />
            ))}
          </div>
        </div>
      )}

      {/* DETAILS */}
      <div className="mt-5 grid gap-3 md:grid-cols-2">
        <CollapsibleList
          title="Repeated Findings"
          items={summary.repeated_findings}
          open
        />

        <CollapsibleList
          title="Suggested Next Steps"
          items={summary.suggestions}
          open
        />

        <CollapsibleList
          title="Common Symptoms"
          items={summary.common_symptoms}
        />

        <CollapsibleList
          title="Questions for a Healthcare Professional"
          items={summary.questions_for_doctor}
        />
      </div>

      {/* DISCLAIMER */}
      <details className="mt-4 rounded-xl border border-slate-200 bg-white px-5 py-4">
        <summary className="cursor-pointer text-sm font-medium text-slate-500">
          Research disclaimer
        </summary>

        <p className="mt-3 text-xs leading-5 text-slate-500">
          {summary.disclaimer}
        </p>
      </details>
    </section>
  );
}

/* =========================================================
   RECORDING TAB
========================================================= */

function RecordingTab({
  recording,
  active,
  onClick,
}: {
  recording: ECGRecordingResult;

  active: boolean;

  onClick: () => void;
}) {
  const ctrhythm = recording.models.find((model) => model.name === "CTRhythm");

  const transformer = recording.models.find(
    (model) => model.name === "CNN + Transformer",
  );

  const result = ctrhythm ?? transformer ?? recording.models[0];

  const prediction = result?.prediction ?? "Unknown";

  return (
    <button
      onClick={onClick}

      className={`min-w-47.5 rounded-xl border px-4 py-3 text-left transition ${
        active
          ? "border-slate-900 bg-slate-900 text-white"
          : "border-slate-200 bg-white hover:border-slate-400"
      } `}
    >
      <div className="flex items-center justify-between gap-3">
        <span
          className={
            active ? "text-xs text-slate-300" : "text-xs text-slate-400"
          }
        >
          ECG {recording.sequence}
        </span>

        <StatusDot prediction={prediction} />
      </div>

      <p
        className={`mt-1 truncate text-sm font-semibold ${
          active ? "text-white" : "text-slate-800"
        } `}
      >
        {recording.filename}
      </p>

      <p
        className={`mt-1 text-xs ${
          active ? "text-slate-300" : "text-slate-500"
        } `}
      >
        CTRhythm: {prediction}
      </p>
    </button>
  );
}

/* =========================================================
   MODEL SNAPSHOT
========================================================= */

function ModelSnapshot({ model }: { model: ModelResult }) {
  return (
    <div className="rounded-xl bg-slate-50 p-4">
      <div className="flex items-center justify-between gap-2">
        <p className="truncate text-xs text-slate-400">{model.name}</p>

        <StatusDot prediction={model.prediction} />
      </div>

      <div className="mt-2 flex items-end justify-between gap-3">
        <p className="text-xl font-bold text-slate-900">{model.prediction}</p>

        <p className="text-sm font-semibold text-slate-500">
          {(model.confidence * 100).toFixed(1)}%
        </p>
      </div>
    </div>
  );
}

/* =========================================================
   MODEL CARD
========================================================= */

function ModelCard({ result }: { result: ModelResult }) {
  const metrics = modelMetrics[result.name];

  const abnormal = result.prediction !== "Normal";

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      {/* TOP */}
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <p className="font-semibold text-slate-800">{result.name}</p>

          {result.name === "CTRhythm" && (
            <span className="rounded-full bg-slate-900 px-2 py-1 text-[10px] font-semibold tracking-wide text-white uppercase">
              Paper Model
            </span>
          )}
        </div>

        {abnormal && (
          <span className="rounded-full bg-amber-50 px-2.5 py-1 text-xs font-medium text-amber-700">
            Review
          </span>
        )}
      </div>

      {/* MAIN RESULT */}
      <div className="mt-5 flex items-end justify-between gap-4">
        <div>
          <p className="text-xs tracking-wide text-slate-400 uppercase">
            Prediction
          </p>

          <p className="mt-1 text-3xl font-bold text-slate-900">
            {result.prediction}
          </p>
        </div>

        <div className="text-right">
          <p className="text-xs text-slate-400">Confidence</p>

          <p className="mt-1 text-xl font-semibold text-slate-800">
            {(result.confidence * 100).toFixed(1)}%
          </p>
        </div>
      </div>

      {/* CONFIDENCE BAR */}
      <div className="mt-4 h-2 overflow-hidden rounded-full bg-slate-100">
        <div
          className="h-full rounded-full bg-slate-900"

          style={{
            width: `${result.confidence * 100}%`,
          }}
        />
      </div>

      {/* PROBABILITIES */}
      <div className="mt-5 grid grid-cols-2 gap-3">
        {Object.entries(result.probabilities).map(([label, probability]) => (
          <CompactProbability
            key={label}

            label={label}

            probability={probability}
          />
        ))}
      </div>

      {/* METRICS */}
      {metrics && (
        <div
          className={`mt-5 grid gap-3 border-t border-slate-100 pt-4 ${
            metrics.paperF1 !== undefined
              ? "grid-cols-2 xl:grid-cols-4"
              : "grid-cols-3"
          } `}
        >
          <SmallStat
            label="Accuracy"

            value={`${(metrics.accuracy * 100).toFixed(1)}%`}
          />

          <SmallStat
            label="Macro F1"

            value={metrics.macroF1.toFixed(3)}
          />

          {metrics.paperF1 !== undefined && (
            <SmallStat
              label="Paper F1"

              value={metrics.paperF1.toFixed(3)}
            />
          )}

          <SmallStat
            label="Runtime"

            value={`${result.inference_ms.toFixed(1)}ms`}
          />
        </div>
      )}

      {/* CTRHYTHM NOTE */}
      {result.name === "CTRhythm" && (
        <p className="mt-4 text-[11px] leading-5 text-slate-400">
          CTRhythm metrics shown here are from the best Fold 1 validation
          checkpoint.
        </p>
      )}
    </div>
  );
}

/* =========================================================
   CONDITION CARD
========================================================= */

function ConditionCard({
  name,
  relevance,
  reason,
}: {
  name: string;

  relevance: "Higher" | "Possible" | "Lower";

  reason: string;
}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4">
      <div className="flex items-start justify-between gap-3">
        <p className="font-semibold text-slate-800">{name}</p>

        <span className="shrink-0 rounded-full bg-slate-100 px-2 py-1 text-[11px] text-slate-500">
          {relevance}
        </span>
      </div>

      <p className="mt-2 text-xs leading-5 text-slate-500">{reason}</p>
    </div>
  );
}

/* =========================================================
   COLLAPSIBLE LIST
========================================================= */

function CollapsibleList({
  title,
  items,
  open = false,
}: {
  title: string;

  items: string[];

  open?: boolean;
}) {
  return (
    <details
      open={open}

      className="rounded-xl border border-slate-200 bg-white px-5 py-4"
    >
      <summary className="cursor-pointer list-none">
        <div className="flex items-center justify-between gap-3">
          <span className="font-semibold text-slate-800">{title}</span>

          <span className="rounded-full bg-slate-100 px-2 py-1 text-xs text-slate-500">
            {items.length}
          </span>
        </div>
      </summary>

      <div className="mt-4 space-y-3">
        {items.length > 0 ? (
          items.map((item, index) => (
            <div
              key={`${item}-${index}`}

              className="flex items-start gap-3"
            >
              <span className="mt-1 flex h-5 w-5 shrink-0 items-center justify-center rounded-md bg-slate-100 text-[10px] font-semibold text-slate-500">
                {index + 1}
              </span>

              <p className="text-sm leading-6 text-slate-600">{item}</p>
            </div>
          ))
        ) : (
          <p className="text-sm text-slate-400">None listed.</p>
        )}
      </div>
    </details>
  );
}

/* =========================================================
   URGENCY BADGE
========================================================= */

function UrgencyBadge({ urgency }: { urgency: HistoryAISummary["urgency"] }) {
  if (urgency === "Urgent if symptoms are present") {
    return (
      <span className="inline-flex items-center gap-2 rounded-full bg-red-50 px-3 py-1.5 text-xs font-semibold text-red-700">
        <span className="h-2 w-2 rounded-full bg-red-500" />
        Higher attention
      </span>
    );
  }

  if (urgency === "Consider medical review") {
    return (
      <span className="inline-flex items-center gap-2 rounded-full bg-amber-50 px-3 py-1.5 text-xs font-semibold text-amber-700">
        <span className="h-2 w-2 rounded-full bg-amber-500" />
        Review suggested
      </span>
    );
  }

  return (
    <span className="inline-flex items-center gap-2 rounded-full bg-emerald-50 px-3 py-1.5 text-xs font-semibold text-emerald-700">
      <span className="h-2 w-2 rounded-full bg-emerald-500" />
      Routine
    </span>
  );
}

/* =========================================================
   STATUS DOT
========================================================= */

function StatusDot({ prediction }: { prediction: string }) {
  const style =
    prediction === "Normal"
      ? "bg-emerald-500"
      : prediction === "Noisy"
        ? "bg-slate-400"
        : "bg-amber-500";

  return <span className={`h-2 w-2 shrink-0 rounded-full ${style}`} />;
}

/* =========================================================
   TAB BUTTON
========================================================= */

function TabButton({
  children,
  active,
  onClick,
}: {
  children: React.ReactNode;

  active: boolean;

  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}

      className={`rounded-lg px-4 py-2 text-sm font-medium transition ${
        active
          ? "bg-slate-900 text-white"
          : "text-slate-500 hover:text-slate-900"
      } `}
    >
      {children}
    </button>
  );
}

/* =========================================================
   MINI METRIC
========================================================= */

function MiniMetric({
  label,
  value,
}: {
  label: string;

  value: string;
}) {
  return (
    <div>
      <p className="text-[11px] tracking-wide text-slate-400 uppercase">
        {label}
      </p>

      <p className="mt-0.5 text-sm font-semibold text-slate-700">{value}</p>
    </div>
  );
}

/* =========================================================
   PROBABILITY
========================================================= */

function CompactProbability({
  label,
  probability,
}: {
  label: string;

  probability: number;
}) {
  return (
    <div className="rounded-lg bg-slate-50 px-3 py-2">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs text-slate-500">{label}</span>

        <span className="text-xs font-semibold text-slate-700">
          {(probability * 100).toFixed(1)}%
        </span>
      </div>
    </div>
  );
}

/* =========================================================
   SMALL STAT
========================================================= */

function SmallStat({
  label,
  value,
}: {
  label: string;

  value: string;
}) {
  return (
    <div>
      <p className="text-[10px] tracking-wide text-slate-400 uppercase">
        {label}
      </p>

      <p className="mt-1 text-sm font-semibold text-slate-700">{value}</p>
    </div>
  );
}

/* =========================================================
   COMPACT TEXT
========================================================= */

function CompactText({
  title,
  text,
}: {
  title: string;

  text: string;
}) {
  return (
    <div>
      <p className="text-sm font-semibold text-slate-800">{title}</p>

      <p className="mt-1 text-sm leading-6 text-slate-500">{text}</p>
    </div>
  );
}
