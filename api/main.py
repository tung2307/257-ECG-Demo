from io import BytesIO
from pathlib import Path
from typing import Literal

import json
import math
import os
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
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
        "https://257-ecg-demo.vercel.app",
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
# SETTINGS
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


DEVICE = torch.device(
    "cpu"
)


# =========================================================
# OFFICIAL CTRHYTHM ARCHITECTURE
# Embedded so Render does not need a local model.py
# =========================================================

class ConvBlock(nn.Module):

    def __init__(
        self,
        channel_in,
        channel,
        stride=1,
    ):

        super().__init__()

        self.conv = nn.Conv1d(
            channel_in,
            channel,
            kernel_size=3,
            padding=1,
            stride=stride,
        )

        self.norm = nn.BatchNorm1d(
            channel
        )

        self.activate = nn.ReLU()


    def forward(
        self,
        x,
    ):

        x = self.conv(
            x
        )

        x = self.norm(
            x
        )

        x = self.activate(
            x
        )

        return x


class CTRhythmResidualBlock(nn.Module):

    def __init__(
        self,
        channel_in=32,
        channel=32,
        stride=1,
        dropout=0.3,
    ):

        super().__init__()

        self.stride = stride

        self.dropout = nn.Dropout(
            p=dropout,
            inplace=False,
        )


        self.main = nn.Sequential(

            ConvBlock(
                channel_in,
                channel,
                stride=stride,
            ),

            ConvBlock(
                channel,
                channel,
            ),

            ConvBlock(
                channel,
                channel,
            ),
        )


        self.skip = None


        if self.stride == 2:

            self.skip = nn.Conv1d(
                channel_in,
                channel,
                kernel_size=3,
                padding=1,
                stride=stride,
            )


    def forward(
        self,
        x,
    ):

        x = self.dropout(
            x
        )

        y = self.main(
            x
        )


        if self.stride == 2:

            x = self.skip(
                x
            )


        x = (
            y
            +
            x
        )

        return x


class CNNModule(nn.Module):

    def __init__(
        self,
        channel_in=1,
        downsample_depth=5,
        main_depth=1,
        dropout=0.3,
    ):

        super().__init__()


        self.downsample_layers = nn.Sequential(

            CTRhythmResidualBlock(
                channel_in,
                16,
                2,
                dropout,
            ),

            CTRhythmResidualBlock(
                16,
                32,
                2,
                dropout,
            ),
        )


        self.main = nn.Sequential(

            CTRhythmResidualBlock(
                dropout=dropout
            )
        )


        if downsample_depth > 2:

            for _ in range(
                downsample_depth
                -
                2
            ):

                self.downsample_layers.append(

                    CTRhythmResidualBlock(
                        32,
                        32,
                        2,
                        dropout=dropout,
                    )
                )


        if main_depth > 1:

            for _ in range(
                main_depth
                -
                1
            ):

                self.downsample_layers.append(

                    CTRhythmResidualBlock(
                        32,
                        32,
                        1,
                        dropout=dropout,
                    )
                )


    def forward(
        self,
        x,
    ):

        x = self.downsample_layers(
            x
        )

        x = self.main(
            x
        )

        return x


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_in,
        d_model,
        dropout=0.1,
        max_len=10000,
    ):

        super().__init__()

        self.dropout = nn.Dropout(
            p=dropout
        )


        position = (
            torch.arange(
                max_len
            )
            .unsqueeze(
                1
            )
        )


        div_term = torch.exp(

            torch.arange(
                0,
                d_model,
                2,
            )

            *

            -(
                math.log(
                    10000.0
                )
                /
                d_model
            )
        )


        pe = torch.zeros(
            max_len,
            1,
            d_model,
        )


        pe[
            :,
            0,
            0::2,
        ] = torch.sin(
            position
            *
            div_term
        )


        pe[
            :,
            0,
            1::2,
        ] = torch.cos(
            position
            *
            div_term
        )


        self.register_buffer(
            "pe",
            pe,
        )


        self.linear = nn.Linear(
            d_in,
            d_model,
        )


    def forward(
        self,
        x,
    ):

        x = self.linear(
            x
        )


        x = (
            x
            +
            self.pe[
                :x.size(0),
                :
            ]
        )


        x = self.dropout(
            x
        )

        return x


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        dropout=0.1,
    ):

        super().__init__()


        encoder_layer = (
            nn.TransformerEncoderLayer(

                d_model,

                nhead,

                dim_feedforward,

                dropout,

                batch_first=True,
            )
        )


        self.transformer_encoder = (
            nn.TransformerEncoder(

                encoder_layer,

                num_layers,
            )
        )


    def forward(
        self,
        x,
    ):

        x = (
            self.transformer_encoder(
                x
            )
        )

        return x


class CTRhythm(nn.Module):

    def __init__(
        self,
        init_channels=1,
        d_model=32,
        nhead=8,
        num_layers=6,
        dim_feedforward=256,
        num_classes=4,
        dropout=0.1,
    ):

        super().__init__()


        self.CNN = (
            CNNModule(

                channel_in=
                    init_channels,

                downsample_depth=
                    5,

                main_depth=
                    1,

                dropout=
                    0.3,
            )
        )


        self.pos_encoding = (
            PositionalEncoding(

                32,

                d_model,

                dropout,
            )
        )


        self.transformer_encoder = (
            TransformerEncoder(

                d_model,

                nhead,

                num_layers,

                dim_feedforward,

                dropout=
                    dropout,
            )
        )


        self.classification = (
            nn.Linear(

                d_model,

                num_classes,
            )
        )


    def forward_feature(
        self,
        x,
    ):

        x = self.CNN(
            x
        )


        x = x.permute(
            2,
            0,
            1,
        )


        x = self.pos_encoding(
            x
        )


        x = x.permute(
            1,
            0,
            2,
        )


        x = (
            self.transformer_encoder(
                x
            )
        )


        x = torch.mean(
            x,
            dim=1,
        )


        return x


    def forward(
        self,
        x,
    ):

        x = self.forward_feature(
            x
        )


        x = self.classification(
            x
        )


        return x


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

        identity = self.shortcut(
            x
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
# CHECKPOINT HELPER
# =========================================================

def get_state_dict(
    checkpoint,
):

    if (
        isinstance(
            checkpoint,
            dict,
        )
        and
        "model_state_dict"
        in checkpoint
    ):

        checkpoint = (
            checkpoint[
                "model_state_dict"
            ]
        )


    if not isinstance(
        checkpoint,
        dict,
    ):

        raise ValueError(
            "Checkpoint does not contain a state_dict."
        )


    if (
        checkpoint
        and
        all(
            key.startswith(
                "module."
            )
            for key
            in checkpoint.keys()
        )
    ):

        checkpoint = {

            key.removeprefix(
                "module."
            ):
                value

            for (
                key,
                value,
            )
            in checkpoint.items()
        }


    return checkpoint


# =========================================================
# LOAD MODELS
# =========================================================

def load_models():

    missing = [

        str(
            path
        )

        for path
        in [
            RF_PATH,
            TRANSFORMER_PATH,
            CTRHYTHM_PATH,
        ]

        if not path.exists()
    ]


    if missing:

        raise FileNotFoundError(

            "Missing model file(s): "
            +
            ", ".join(
                missing
            )
        )


    # -----------------------------------------------------
    # RANDOM FOREST
    # -----------------------------------------------------

    rf = joblib.load(
        RF_PATH
    )


    # -----------------------------------------------------
    # CNN + TRANSFORMER
    # -----------------------------------------------------

    transformer = (
        ECGCNNTransformer(
            num_classes=4
        )
    )


    transformer_checkpoint = (
        torch.load(

            TRANSFORMER_PATH,

            map_location=
                DEVICE,

            weights_only=
                False,
        )
    )


    transformer_state = (
        get_state_dict(
            transformer_checkpoint
        )
    )


    transformer.load_state_dict(
        transformer_state
    )


    transformer.to(
        DEVICE
    )


    transformer.eval()


    # -----------------------------------------------------
    # CTRHYTHM
    # -----------------------------------------------------

    ctrhythm = CTRhythm(

        init_channels=1,

        d_model=32,

        nhead=8,

        num_layers=6,

        dim_feedforward=256,

        num_classes=4,

        dropout=0.1,
    )


    ctrhythm_checkpoint = (
        torch.load(

            CTRHYTHM_PATH,

            map_location=
                DEVICE,

            weights_only=
                False,
        )
    )


    ctrhythm_state = (
        get_state_dict(
            ctrhythm_checkpoint
        )
    )


    ctrhythm.load_state_dict(
        ctrhythm_state
    )


    ctrhythm.to(
        DEVICE
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
    "CTRhythm architecture: embedded in api/main.py"
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


    # PhysioNet / CinC 2017
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


        signal = value.squeeze()


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
        ) from error


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
                filtered[
                    -1
                ]
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

            errors=
                "coerce",
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
    # DETECT SAMPLE RATE FROM TIME COLUMN
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
            "time"
            in name
            or
            "second"
            in name
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

                errors=
                    "coerce",
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
                    differences
                    >
                    0
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
    target_fs=TARGET_FS,
    duration=TARGET_DURATION,
):

    signal = np.asarray(

        signal,

        dtype=
            np.float64,

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
    # RESAMPLE
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
    # PAD / TRUNCATE
    # -----------------------------------------------------

    target_length = int(

        target_fs
        *
        duration
    )


    result = np.zeros(

        target_length,

        dtype=
            np.float32,
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

            dtype=
                np.float32,

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
        )[
            0
        ]
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

        dtype=
            np.float64,
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
                <=
                class_id
                <
                len(
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
            DEVICE
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

            )[
                0
            ]
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
            DEVICE
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

            )[
                0
            ]
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
You are generating a concise educational summary of one or more
single-lead ECG recordings analyzed by research models.

Each ECG is analyzed by three models:

1. Random Forest
2. CNN + Transformer
3. CTRhythm

The models classify:

- Normal
- AFib
- Other
- Noisy

The models are research models and are not clinically validated.

Do not diagnose the person.

Never say the person definitely has a disease.

Use wording such as:

- possible condition to discuss
- may be worth evaluating
- could be considered

Model confidence is not clinical certainty.

CTRhythm Fold 1 project validation:

Accuracy: 0.8658
Normal F1: 0.9173
AFib F1: 0.8121
Other F1: 0.7987
Noisy F1: 0.5556
Paper-style F1: 0.8427
4-class Macro F1: 0.7709

These are research validation results and are not clinical accuracy.

Do not invent recording dates.

Do not invent symptoms.

Only symptoms present in reported_symptoms were actually reported.

If models disagree, clearly explain the disagreement.

Repeated abnormal classifications may be worth discussing with a
healthcare professional, but do not call them a diagnosis.

Possible rhythm-related conditions to discuss can include:

- atrial fibrillation
- another atrial arrhythmia
- premature atrial contractions
- premature ventricular contractions
- another rhythm abnormality
- recording artifact
- poor ECG signal

Do not generate unrelated disease lists.

If reported symptoms include severe chest pain, fainting,
severe shortness of breath, facial drooping, difficulty speaking,
or new one-sided weakness, urgent medical evaluation may be
appropriate.

Keep the summary concise and easy to understand.

The disclaimer must state that this output is for educational and
research use only, is not a medical diagnosis, and these models are
not clinically validated for general medical use.
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
            "embedded in api/main.py",

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


        except Exception as error:

            raise HTTPException(
                status_code=400,
                detail="sample_rates must be a JSON list.",
            ) from error


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
                    <
                    len(
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