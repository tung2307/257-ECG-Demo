export type Probabilities = {
  Normal: number;
  AFib: number;
  Other: number;
  Noisy: number;
};

export type ModelResult = {
  name: string;
  prediction: string;
  confidence: number;
  probabilities: Probabilities;
  inference_ms: number;
};

export type SignalInfo = {
  original_samples: number;
  original_sample_rate: number;
  duration_seconds: number;
  processed_samples: number;
  sample_rate: number;
};

export type ECGRecordingResult = {
  sequence: number;
  filename: string;
  file_type: string;
  signal: SignalInfo;
  models: ModelResult[];
};

export type FileError = {
  sequence: number;
  filename: string;
  error: string;
};

export type PossibleCondition = {
  name: string;

  relevance: "Higher" | "Possible" | "Lower";

  reason: string;
};

export type HistoryAISummary = {
  overview: string;

  trend: string;

  model_agreement: string;

  repeated_findings: string[];

  possible_conditions: PossibleCondition[];

  common_symptoms: string[];

  suggestions: string[];

  questions_for_doctor: string[];

  urgency:
    "Routine" | "Consider medical review" | "Urgent if symptoms are present";

  disclaimer: string;
};

export type MultiECGResponse = {
  recording_count: number;

  failed_count: number;

  recordings: ECGRecordingResult[];

  file_errors: FileError[];

  ai_summary: HistoryAISummary | null;

  ai_error: string | null;
};
