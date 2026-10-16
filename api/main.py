from io import BytesIO
from pathlib import Path
from typing import Literal

import importlib.util
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
)

from fastapi.middleware.cors import CORSMiddleware

from openai import OpenAI
from pydantic import BaseModel

from scipy.io import loadmat
from sklearn.preprocessing import scale


# =========================================================
# ENV
# =========================================================

load_dotenv()

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-6-astra",
)

openai_client = (
    OpenAI(
        api_key=OPENAI_API_KEY
    )
    if OPENAI_API_KEY
    else None
)


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="ECG Rhythm API",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:3000",
    ],

    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

MODEL_DIR = (
    BASE_DIR
    / "models"
)


RF_PATH = (
    MODEL_DIR
    / "ecg_random_forest.pkl"
)


TRANSFORMER_PATH = (
    MODEL_DIR
    / "ecg_cnn_transformer_best.pth"
)


CTRHYTHM_PATH = (
    MODEL_DIR
    / "ctrhythm_fold1_best.pth"
)


# =========================================================
# OFFICIAL CTRHYTHM SOURCE
# =========================================================

CTRHYTHM_DIR = (
    Path.home()
    / "Downloads"
    / "CMPE257"
    / "CTRhythm"
)

CTRHYTHM_SOURCE = (
    CTRHYTHM_DIR
    / "model.py"
)


if not CTRHYTHM_SOURCE.exists():

    raise FileNotFoundError(
        f"CTRhythm model.py not found: {CTRHYTHM_SOURCE}"
    )


# Allows model.py to find any local CTRhythm files
if str(CTRHYTHM_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(CTRHYTHM_DIR),
    )


spec = (
    importlib.util
    .spec_from_file_location(
        "official_ctrhythm_model",
        CTRHYTHM_SOURCE,
    )
)


if (
    spec is None
    or
    spec.loader is None
):

    raise ImportError(
        f"Could not load CTRhythm model from {CTRHYTHM_SOURCE}"
    )


ctrhythm_module = (
    importlib.util
    .module_from_spec(
        spec
    )
)


spec.loader.exec_module(
    ctrhythm_module
)


CTRhythm = (
    ctrhythm_module
    .CTRhythm
)


# =========================================================
# ECG SETTINGS
# =========================================================

CLASS_NAMES = [
    "Normal",
    "AFib",
    "Other",
    "Noisy",
]


TARGET_FS = 150
TARGET_DURATION = 60
TARGET_LENGTH = (
    TARGET_FS
    *
    TARGET_DURATION
)


# CPU is easiest/stable for API inference
device = torch.device(
    "cpu"
)


# =========================================================
# SIMPLIFIED CNN + TRANSFORMER
# =========================================================

class ResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
    ):

        super().__init__()


        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=5,
            stride=stride,
            padding=2,
        )


        self.bn1 = nn.BatchNorm1d(
            out_channels
        )


        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )


        self.bn2 = nn.BatchNorm1d(
            out_channels
        )


        self.relu = nn.ReLU()


        self.shortcut = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
        )


    def forward(
        self,
        x,
    ):

        identity = (
            self.shortcut(
                x
            )
        )


        x = self.relu(
            self.bn1(
                self.conv1(
                    x
                )
            )
        )


        x = self.bn2(
            self.conv2(
                x
            )
        )


        x = (
            x
            +
            identity
        )


        x = self.relu(
            x
        )


        return x


class ECGCNNTransformer(nn.Module):

    def __init__(
        self,
        num_classes=4,
    ):

        super().__init__()


        self.cnn = nn.Sequential(

            ResidualBlock(
                1,
                32,
            ),

            ResidualBlock(
                32,
                32,
            ),

            ResidualBlock(
                32,
                64,
            ),

            ResidualBlock(
                64,
                64,
            ),

            ResidualBlock(
                64,
                128,
            ),

            ResidualBlock(
                128,
                128,
            ),
        )


        self.pos_embedding = nn.Parameter(
            torch.zeros(
                1,
                141,
                128,
            )
        )


        encoder_layer = (
            nn.TransformerEncoderLayer(

                d_model=128,

                nhead=4,

                dim_feedforward=256,

                dropout=0.1,

                batch_first=True,
            )
        )


        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=2,
            )
        )


        self.classifier = nn.Sequential(

            nn.AdaptiveAvgPool1d(
                1
            ),

            nn.Flatten(),

            nn.Dropout(
                0.2
            ),

            nn.Linear(
                128,
                num_classes,
            ),
        )


    def forward(
        self,
        x,
    ):

        x = self.cnn(
            x
        )


        x = x.transpose(
            1,
            2,
        )


        x = (
            x
            +
            self.pos_embedding[
                :,
                :x.size(1),
                :
            ]
        )


        x = self.transformer(
            x
        )


        x = x.transpose(
            1,
            2,
        )


        return self.classifier(
            x
        )


# =========================================================
# LOAD MODELS
# =========================================================

def load_models():

    # -----------------------------------------------------
    # FILE CHECK
    # -----------------------------------------------------

    if not RF_PATH.exists():

        raise FileNotFoundError(
            f"Random Forest not found: {RF_PATH}"
        )


    if not TRANSFORMER_PATH.exists():

        raise FileNotFoundError(
            f"CNN Transformer not found: {TRANSFORMER_PATH}"
        )


    if not CTRHYTHM_PATH.exists():

        raise FileNotFoundError(
            f"CTRhythm checkpoint not found: {CTRHYTHM_PATH}"
        )


    # -----------------------------------------------------
    # RANDOM FOREST
    # -----------------------------------------------------

    rf = joblib.load(
        RF_PATH
    )


    # -----------------------------------------------------
    # SIMPLIFIED CNN + TRANSFORMER
    # -----------------------------------------------------

    transformer = (
        ECGCNNTransformer(
            num_classes=4
        )
    )


    transformer_checkpoint = (
        torch.load(
            TRANSFORMER_PATH,
            map_location=device,
            weights_only=False,
        )
    )


    if (
        isinstance(
            transformer_checkpoint,
            dict,
        )
        and
        "model_state_dict"
        in transformer_checkpoint
    ):

        transformer_state = (
            transformer_checkpoint[
                "model_state_dict"
            ]
        )

    else:

        transformer_state = (
            transformer_checkpoint
        )


    transformer.load_state_dict(
        transformer_state
    )


    transformer.to(
        device
    )


    transformer.eval()


    # -----------------------------------------------------
    # CTRHYTHM CHECKPOINT
    # -----------------------------------------------------

    ctrhythm_checkpoint = (
        torch.load(
            CTRHYTHM_PATH,
            map_location=device,
            weights_only=False,
        )
    )


    # These are the settings used for your training
    ctrhythm = CTRhythm(

        init_channels=1,

        d_model=32,

        nhead=8,

        num_layers=6,

        dim_feedforward=256,

        num_classes=4,

        dropout=0.1,
    )


    if (
        isinstance(
            ctrhythm_checkpoint,
            dict,
        )
        and
        "model_state_dict"
        in ctrhythm_checkpoint
    ):

        ctrhythm_state = (
            ctrhythm_checkpoint[
                "model_state_dict"
            ]
        )

    else:

        ctrhythm_state = (
            ctrhythm_checkpoint
        )


    ctrhythm.load_state_dict(
        ctrhythm_state
    )


    ctrhythm.to(
        device
    )


    ctrhythm.eval()


    return (
        rf,
        transformer,
        ctrhythm,
    )


(
    rf_model,
    transformer_model,
    ctrhythm_model,
) = load_models()


print(
    "========================================"
)

print(
    "Random Forest loaded"
)

print(
    "CNN + Transformer loaded"
)

print(
    "CTRhythm loaded"
)

print(
    f"CTRhythm architecture: {CTRHYTHM_SOURCE}"
)

print(
    f"CTRhythm checkpoint: {CTRHYTHM_PATH}"
)

print(
    "========================================"
)


# =========================================================
# MAT READER
# =========================================================

def read_mat_ecg(
    content: bytes,
):

    mat = loadmat(
        BytesIO(
            content
        )
    )


    # CINC 2017 format
    if "val" in mat:

        signal = (
            mat[
                "val"
            ]
            .squeeze()
        )


        return (
            signal.astype(
                np.float64
            ),
            300.0,
        )


    # Fallback
    for (
        key,
        value,
    ) in mat.items():

        if key.startswith(
            "__"
        ):

            continue


        if not isinstance(
            value,
            np.ndarray,
        ):

            continue


        if not np.issubdtype(
            value.dtype,
            np.number,
        ):

            continue


        signal = (
            value.squeeze()
        )


        if (
            signal.ndim == 1
            and
            signal.size > 100
        ):

            return (
                signal.astype(
                    np.float64
                ),
                300.0,
            )


    raise ValueError(
        "Could not find ECG signal in MAT file."
    )


# =========================================================
# CSV READER
# =========================================================

def read_csv_ecg(
    content: bytes,
    provided_sample_rate=None,
):

    try:

        df = pd.read_csv(
            BytesIO(
                content
            )
        )

    except Exception as error:

        raise ValueError(
            f"Could not read CSV: {error}"
        )


    if df.empty:

        raise ValueError(
            "CSV file is empty."
        )


    preferred_names = [
        "ecg",
        "signal",
        "voltage",
        "value",
        "amplitude",
        "mv",
        "microvolt",
        "microvolts",
    ]


    signal_column = None


    for column in df.columns:

        name = (
            str(
                column
            )
            .strip()
            .lower()
        )


        if any(
            word in name
            for word
            in preferred_names
        ):

            signal_column = (
                column
            )

            break


    # -----------------------------------------------------
    # FALLBACK NUMERIC COLUMN
    # -----------------------------------------------------

    if signal_column is None:

        numeric_columns = (
            df.select_dtypes(
                include=[
                    np.number
                ]
            )
            .columns
            .tolist()
        )


        ignored = [
            "time",
            "timestamp",
            "seconds",
            "second",
            "sample",
            "index",
        ]


        filtered = []


        for column in numeric_columns:

            name = (
                str(
                    column
                )
                .strip()
                .lower()
            )


            if not any(
                word in name
                for word
                in ignored
            ):

                filtered.append(
                    column
                )


        if filtered:

            signal_column = (
                filtered[-1]
            )


    if signal_column is None:

        if len(
            df.columns
        ) == 1:

            signal_column = (
                df.columns[
                    0
                ]
            )

        else:

            raise ValueError(
                "Could not determine ECG signal column."
            )


    signal = (
        pd.to_numeric(
            df[
                signal_column
            ],
            errors="coerce",
        )
        .dropna()
        .to_numpy(
            dtype=np.float64
        )
    )


    if signal.size == 0:

        raise ValueError(
            "CSV contains no valid ECG samples."
        )


    # -----------------------------------------------------
    # PROVIDED SAMPLE RATE
    # -----------------------------------------------------

    if provided_sample_rate is not None:

        sample_rate = float(
            provided_sample_rate
        )


        if sample_rate <= 0:

            raise ValueError(
                "Sampling rate must be greater than 0."
            )


        return (
            signal,
            sample_rate,
        )


    # -----------------------------------------------------
    # DETECT FROM TIME COLUMN
    # -----------------------------------------------------

    time_column = None


    for column in df.columns:

        name = (
            str(
                column
            )
            .strip()
            .lower()
        )


        if (
            "time" in name
            or
            "second" in name
        ):

            time_column = (
                column
            )

            break


    if time_column is not None:

        times = (
            pd.to_numeric(
                df[
                    time_column
                ],
                errors="coerce",
            )
            .dropna()
            .to_numpy(
                dtype=np.float64
            )
        )


        if len(
            times
        ) > 2:

            differences = np.diff(
                times
            )


            differences = (
                differences[
                    differences > 0
                ]
            )


            if len(
                differences
            ) > 0:

                median_delta = (
                    np.median(
                        differences
                    )
                )


                if median_delta > 0:

                    return (
                        signal,
                        1.0
                        /
                        median_delta,
                    )


    raise ValueError(
        "Could not determine CSV sampling rate. "
        "Enter the original sampling rate."
    )


# =========================================================
# PREPROCESSING
# =========================================================

def preprocess_ecg(
    signal,
    original_fs,
    target_fs=150,
    duration=60,
):

    signal = np.asarray(
        signal,
        dtype=np.float64,
    ).squeeze()


    if (
        signal.ndim != 1
        or
        signal.size == 0
    ):

        raise ValueError(
            "Invalid ECG signal."
        )


    if not np.isfinite(
        signal
    ).all():

        raise ValueError(
            "ECG contains invalid values."
        )


    if signal.std() < 1e-8:

        raise ValueError(
            "ECG signal is flat."
        )


    if original_fs <= 0:

        raise ValueError(
            "Invalid sampling rate."
        )


    # -----------------------------------------------------
    # SAME RESAMPLING USED DURING TRAINING
    # -----------------------------------------------------

    if original_fs != target_fs:

        signal = np.interp(

            np.arange(
                0,
                len(
                    signal
                ),
                original_fs
                /
                target_fs,
            ),

            np.arange(
                len(
                    signal
                )
            ),

            signal,
        )


    # -----------------------------------------------------
    # Z-SCORE
    # -----------------------------------------------------

    signal = scale(
        signal
    )


    # -----------------------------------------------------
    # PAD / TRUNCATE TO 9000
    # -----------------------------------------------------

    target_length = int(
        target_fs
        *
        duration
    )


    result = np.zeros(
        target_length,
        dtype=np.float32,
    )


    n = min(
        len(
            signal
        ),
        target_length,
    )


    result[
        :n
    ] = signal[
        :n
    ]


    return result


# =========================================================
# RANDOM FOREST FEATURES
# =========================================================

def extract_rf_features(
    signal,
    segments=30,
):

    parts = np.array_split(
        signal,
        segments,
    )


    features = []


    for part in parts:

        features.extend(
            [
                np.mean(
                    part
                ),

                np.std(
                    part
                ),

                np.min(
                    part
                ),

                np.max(
                    part
                ),
            ]
        )


    return (
        np.array(
            features,
            dtype=np.float32,
        )
        .reshape(
            1,
            -1,
        )
    )


# =========================================================
# RANDOM FOREST PREDICTION
# =========================================================

def predict_random_forest(
    signal,
):

    features = (
        extract_rf_features(
            signal
        )
    )


    start = (
        time.perf_counter()
    )


    raw_probabilities = (
        rf_model.predict_proba(
            features
        )[0]
    )


    elapsed_ms = (
        time.perf_counter()
        -
        start
    ) * 1000


    probabilities = np.zeros(
        len(
            CLASS_NAMES
        ),
        dtype=np.float64,
    )


    if hasattr(
        rf_model,
        "classes_",
    ):

        for (
            model_index,
            class_id,
        ) in enumerate(
            rf_model.classes_
        ):

            class_id = int(
                class_id
            )


            if (
                0
                <= class_id
                < len(
                    CLASS_NAMES
                )
            ):

                probabilities[
                    class_id
                ] = (
                    raw_probabilities[
                        model_index
                    ]
                )


    else:

        probabilities[
            :len(
                raw_probabilities
            )
        ] = raw_probabilities


    prediction_index = int(
        np.argmax(
            probabilities
        )
    )


    return {

        "name":
            "Random Forest",

        "prediction":
            CLASS_NAMES[
                prediction_index
            ],

        "confidence":
            float(
                probabilities[
                    prediction_index
                ]
            ),

        "probabilities": {

            CLASS_NAMES[
                index
            ]:
                float(
                    probabilities[
                        index
                    ]
                )

            for index
            in range(
                len(
                    CLASS_NAMES
                )
            )
        },

        "inference_ms":
            round(
                elapsed_ms,
                2,
            ),
    }


# =========================================================
# CNN + TRANSFORMER PREDICTION
# =========================================================

def predict_transformer(
    signal,
):

    tensor = (
        torch.from_numpy(
            signal
        )
        .float()
        .unsqueeze(
            0
        )
        .unsqueeze(
            0
        )
        .to(
            device
        )
    )


    start = (
        time.perf_counter()
    )


    with torch.no_grad():

        logits = (
            transformer_model(
                tensor
            )
        )


        probabilities = (
            torch.softmax(
                logits,
                dim=1,
            )[0]
        )


    elapsed_ms = (
        time.perf_counter()
        -
        start
    ) * 1000


    probabilities = (
        probabilities
        .cpu()
        .numpy()
    )


    prediction_index = int(
        np.argmax(
            probabilities
        )
    )


    return {

        "name":
            "CNN + Transformer",

        "prediction":
            CLASS_NAMES[
                prediction_index
            ],

        "confidence":
            float(
                probabilities[
                    prediction_index
                ]
            ),

        "probabilities": {

            CLASS_NAMES[
                index
            ]:
                float(
                    probabilities[
                        index
                    ]
                )

            for index
            in range(
                len(
                    CLASS_NAMES
                )
            )
        },

        "inference_ms":
            round(
                elapsed_ms,
                2,
            ),
    }


# =========================================================
# CTRHYTHM PREDICTION
# =========================================================

def predict_ctrhythm(
    signal,
):

    tensor = (
        torch.from_numpy(
            signal
        )
        .float()
        .unsqueeze(
            0
        )
        .unsqueeze(
            0
        )
        .to(
            device
        )
    )


    start = (
        time.perf_counter()
    )


    with torch.no_grad():

        logits = (
            ctrhythm_model(
                tensor
            )
        )


        probabilities = (
            torch.softmax(
                logits,
                dim=1,
            )[0]
        )


    elapsed_ms = (
        time.perf_counter()
        -
        start
    ) * 1000


    probabilities = (
        probabilities
        .cpu()
        .numpy()
    )


    prediction_index = int(
        np.argmax(
            probabilities
        )
    )


    return {

        "name":
            "CTRhythm",

        "prediction":
            CLASS_NAMES[
                prediction_index
            ],

        "confidence":
            float(
                probabilities[
                    prediction_index
                ]
            ),

        "probabilities": {

            CLASS_NAMES[
                index
            ]:
                float(
                    probabilities[
                        index
                    ]
                )

            for index
            in range(
                len(
                    CLASS_NAMES
                )
            )
        },

        "inference_ms":
            round(
                elapsed_ms,
                2,
            ),
    }


# =========================================================
# OPENAI SCHEMA
# =========================================================

class PossibleCondition(
    BaseModel
):

    name: str

    relevance: Literal[
        "Higher",
        "Possible",
        "Lower",
    ]

    reason: str


class HistoryAISummary(
    BaseModel
):

    overview: str

    trend: str

    model_agreement: str

    repeated_findings: list[
        str
    ]

    possible_conditions: list[
        PossibleCondition
    ]

    common_symptoms: list[
        str
    ]

    suggestions: list[
        str
    ]

    questions_for_doctor: list[
        str
    ]

    urgency: Literal[
        "Routine",
        "Consider medical review",
        "Urgent if symptoms are present",
    ]

    disclaimer: str


# =========================================================
# AI INSTRUCTIONS
# =========================================================

AI_INSTRUCTIONS = """
You are generating an educational summary of one or more
single-lead ECG recordings analyzed by research models.

Each ECG is analyzed by three models:

1. Random Forest
2. Simplified CNN + Transformer
3. CTRhythm

Random Forest is the simple baseline.

CNN + Transformer is a simplified deep-learning model
created for this project.

CTRhythm is the implementation of the architecture from
the reference CTRhythm paper.

The current CTRhythm checkpoint is from Fold 1.

Fold 1 validation results:

Accuracy: 0.8658
Normal F1: 0.9173
AFib F1: 0.8121
Other F1: 0.7987
Noisy F1: 0.5556
Paper-style F1: 0.8427
4-class Macro F1: 0.7709

These are research validation results and are not
clinical accuracy.

Do not diagnose the person.

Never say the person definitely has a disease.

Use wording such as:

- possible condition to discuss
- may be worth evaluating
- could be considered

Model confidence is not clinical certainty.

These models are not clinically validated.

Recordings are supplied in upload order.

Do not invent recording dates.

Do not say a condition became medically worse only
because confidence changed.

If multiple recordings repeatedly receive AFib-like
classifications, explain that repeated abnormal model
results may be worth discussing with a healthcare
professional.

If results differ between models, clearly explain the
disagreement.

CTRhythm achieved the strongest paper-style Fold 1
validation result in this project, but do not automatically
assume it is clinically correct.

Never invent symptoms.

Only symptoms present in reported_symptoms were actually
reported.

Possible rhythm-related conditions may include:

- atrial fibrillation
- another atrial arrhythmia
- premature atrial contractions
- premature ventricular contractions
- another rhythm abnormality
- recording artifact
- poor ECG signal

Do not generate unrelated disease lists.

For repeated abnormal classifications, suggestions can
include saving the ECG, showing it to a healthcare
professional, discussing a clinical ECG, or discussing
longer rhythm monitoring.

If reported symptoms include severe chest pain, fainting,
severe shortness of breath, facial drooping, difficulty
speaking, or new one-sided weakness, urgent medical
evaluation may be appropriate.

Keep the summary concise and easy to understand.

The disclaimer must state that this is educational and
research use only, is not a medical diagnosis, and the
models are not clinically validated for general medical use.
"""


# =========================================================
# OPENAI SUMMARY
# =========================================================

def generate_history_summary(
    recordings,
    symptoms=None,
):

    if openai_client is None:

        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )


    data = {

        "recording_count":
            len(
                recordings
            ),

        "model_validation_context": {

            "Random Forest": {
                "accuracy":
                    0.5955,

                "macro_f1":
                    0.2505,
            },

            "CNN + Transformer": {
                "accuracy":
                    0.8535,

                "macro_f1":
                    0.7963,

                "paper_style_f1":
                    0.823,
            },

            "CTRhythm": {
                "fold":
                    1,

                "accuracy":
                    0.8658,

                "paper_style_f1":
                    0.8427,

                "macro_f1":
                    0.7709,
            },
        },

        "recordings":
            recordings,

        "reported_symptoms":
            (
                symptoms
                if symptoms
                else
                "No symptoms reported."
            ),
    }


    response = (
        openai_client
        .responses
        .parse(

            model=
                OPENAI_MODEL,

            input=[
                {
                    "role":
                        "system",

                    "content":
                        AI_INSTRUCTIONS,
                },

                {
                    "role":
                        "user",

                    "content":
                        (
                            "Review these ECG model results.\n\n"
                            +
                            json.dumps(
                                data,
                                indent=2,
                            )
                        ),
                },
            ],

            text_format=
                HistoryAISummary,
        )
    )


    summary = (
        response.output_parsed
    )


    if summary is None:

        raise RuntimeError(
            "OpenAI did not return a structured summary."
        )


    return (
        summary.model_dump()
    )


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():

    return {

        "status":
            "running",

        "version":
            "3.0.0",

        "models": [
            "Random Forest",
            "CNN + Transformer",
            "CTRhythm",
        ],

        "supported_files": [
            ".mat",
            ".csv",
        ],

        "maximum_files":
            12,

        "openai_enabled":
            openai_client
            is not None,
    }


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():

    return {

        "status":
            "ok",

        "random_forest_loaded":
            True,

        "cnn_transformer_loaded":
            True,

        "ctrhythm_loaded":
            True,

        "ctrhythm_architecture":
            str(
                CTRHYTHM_SOURCE
            ),

        "openai_configured":
            openai_client
            is not None,
    }


# =========================================================
# PREDICT
# =========================================================

@app.post("/predict")
async def predict(

    files: list[
        UploadFile
    ] = File(...),

    sample_rates: str | None = Form(
        default=None
    ),

    symptoms: str | None = Form(
        default=None
    ),
):

    # -----------------------------------------------------
    # FILE COUNT
    # -----------------------------------------------------

    if not files:

        raise HTTPException(
            status_code=400,
            detail="Upload at least one ECG file.",
        )


    if len(
        files
    ) > 12:

        raise HTTPException(
            status_code=400,
            detail="Maximum 12 ECG files per analysis.",
        )


    # -----------------------------------------------------
    # SAMPLE RATES
    # -----------------------------------------------------

    parsed_sample_rates = []


    if sample_rates:

        try:

            parsed_sample_rates = (
                json.loads(
                    sample_rates
                )
            )


            if not isinstance(
                parsed_sample_rates,
                list,
            ):

                raise ValueError()


        except Exception:

            raise HTTPException(
                status_code=400,
                detail="sample_rates must be a JSON list.",
            )


    recordings = []
    file_errors = []


    # =====================================================
    # EACH ECG
    # =====================================================

    for (
        index,
        file,
    ) in enumerate(
        files
    ):

        sequence = (
            index
            +
            1
        )


        filename = (
            file.filename
            or
            f"recording_{sequence}"
        )


        extension = (
            Path(
                filename
            )
            .suffix
            .lower()
        )


        if extension not in [
            ".mat",
            ".csv",
        ]:

            file_errors.append({

                "sequence":
                    sequence,

                "filename":
                    filename,

                "error":
                    "Only .mat and .csv files are supported.",
            })

            continue


        try:

            content = (
                await file.read()
            )


            # -------------------------------------------------
            # MAT
            # -------------------------------------------------

            if extension == ".mat":

                (
                    raw_signal,
                    original_fs,
                ) = read_mat_ecg(
                    content
                )


            # -------------------------------------------------
            # CSV
            # -------------------------------------------------

            else:

                supplied_rate = None


                if (
                    index
                    < len(
                        parsed_sample_rates
                    )
                ):

                    supplied_rate = (
                        parsed_sample_rates[
                            index
                        ]
                    )


                (
                    raw_signal,
                    original_fs,
                ) = read_csv_ecg(

                    content,

                    provided_sample_rate=
                        supplied_rate,
                )


            # -------------------------------------------------
            # PREPROCESS
            # -------------------------------------------------

            processed_signal = (
                preprocess_ecg(

                    raw_signal,

                    original_fs=
                        original_fs,

                    target_fs=
                        TARGET_FS,

                    duration=
                        TARGET_DURATION,
                )
            )


            # -------------------------------------------------
            # MODEL 1
            # -------------------------------------------------

            rf_result = (
                predict_random_forest(
                    processed_signal
                )
            )


            # -------------------------------------------------
            # MODEL 2
            # -------------------------------------------------

            transformer_result = (
                predict_transformer(
                    processed_signal
                )
            )


            # -------------------------------------------------
            # MODEL 3
            # -------------------------------------------------

            ctrhythm_result = (
                predict_ctrhythm(
                    processed_signal
                )
            )


            duration_seconds = (
                len(
                    raw_signal
                )
                /
                original_fs
            )


            signal_info = {

                "original_samples":
                    int(
                        len(
                            raw_signal
                        )
                    ),

                "original_sample_rate":
                    round(
                        float(
                            original_fs
                        ),
                        2,
                    ),

                "duration_seconds":
                    round(
                        float(
                            duration_seconds
                        ),
                        2,
                    ),

                "processed_samples":
                    int(
                        len(
                            processed_signal
                        )
                    ),

                "sample_rate":
                    TARGET_FS,
            }


            recordings.append({

                "sequence":
                    sequence,

                "filename":
                    filename,

                "file_type":
                    extension,

                "signal":
                    signal_info,

                "models": [

                    rf_result,

                    transformer_result,

                    ctrhythm_result,
                ],
            })


        except Exception as error:

            print(
                f"Error processing {filename}: {error}"
            )


            file_errors.append({

                "sequence":
                    sequence,

                "filename":
                    filename,

                "error":
                    str(
                        error
                    ),
            })


    # =====================================================
    # NOTHING WORKED
    # =====================================================

    if not recordings:

        raise HTTPException(

            status_code=400,

            detail={

                "message":
                    "No ECG recordings could be analyzed.",

                "file_errors":
                    file_errors,
            },
        )


    # =====================================================
    # AI SUMMARY
    # =====================================================

    ai_summary = None
    ai_error = None


    if openai_client is not None:

        try:

            ai_summary = (
                generate_history_summary(

                    recordings=
                        recordings,

                    symptoms=
                        symptoms,
                )
            )


        except Exception as error:

            print(
                "OpenAI summary error:",
                error,
            )


            ai_error = str(
                error
            )


    else:

        ai_error = (
            "OPENAI_API_KEY is not configured."
        )


    # =====================================================
    # RETURN
    # =====================================================

    return {

        "recording_count":
            len(
                recordings
            ),

        "failed_count":
            len(
                file_errors
            ),

        "recordings":
            recordings,

        "file_errors":
            file_errors,

        "ai_summary":
            ai_summary,

        "ai_error":
            ai_error,
    }