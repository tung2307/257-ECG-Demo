export default function HomePage() {
  return (
    <main className="flex min-h-screen items-center justify-center bg-slate-50 px-6">
      <div className="max-w-2xl text-center">
        <p className="text-sm font-semibold tracking-widest text-slate-400 uppercase">
          CMPE 257
        </p>

        <h1 className="mt-4 text-5xl font-bold tracking-tight text-slate-900">
          ECG Rhythm Analyzer
        </h1>

        <p className="mx-auto mt-6 max-w-xl leading-7 text-slate-500">
          Compare machine-learning and deep-learning ECG rhythm predictions and
          receive an AI-assisted explanation of the results.
        </p>

        <a
          href="/analyze"
          className="mt-8 inline-flex rounded-xl bg-slate-900 px-7 py-4 font-semibold text-white transition hover:bg-slate-800"
        >
          Analyze ECG →
        </a>

        <p className="mt-6 text-xs text-slate-400">
          Research and educational use only.
        </p>
      </div>
    </main>
  );
}
