# ECG Rhythm Analyzer

**CMPE 257 — Machine Learning | San José State University**

An interactive web application for detecting and analyzing atrial fibrillation (AFib) and other cardiac rhythms using machine learning and deep learning models.

The application allows users to upload single-lead electrocardiogram (ECG) recordings, compare predictions from three trained models, and explore AI-assisted interpretations.

**Live Demo:** https://257-ecg-demo.vercel.app/

**Project:** ECG-Based Atrial Fibrillation Detection

## Overview

Atrial fibrillation is a common cardiac arrhythmia characterized by irregular heart rhythm patterns.

This project investigates the effectiveness of traditional machine learning and deep learning approaches for automated ECG classification.

The web application provides an accessible interface for exploring the predictions and performance of three models:

1. Random Forest
2. CNN + Transformer
3. CTRhythm

The application supports four ECG classification categories:

- **Normal:** Normal sinus rhythm
- **AFib:** Atrial fibrillation
- **Other:** Other cardiac rhythm abnormalities
- **Noisy:** Low-quality or inconclusive ECG signals

## Features

### ECG Analysis

- Upload ECG recordings in `.mat` or `.csv` format
- Support for multiple ECG recordings
- Automatic and manual sampling-rate configuration
- ECG signal preprocessing through the backend
- Multi-model inference and prediction comparison

### Model Comparison

- Random Forest baseline
- CNN + Transformer
- CTRhythm research architecture
- Per-model classification confidence
- Class probability distributions
- Model inference runtime
- Validation performance metrics

### AI-Assisted Review

- Summary of model predictions
- Model agreement and disagreement analysis
- Explanation of possible rhythm abnormalities
- Educational recommendations and follow-up questions
- Review of multiple ECG recordings

### User Interface

- Responsive application layout
- Drag-and-drop ECG upload
- Interactive model comparison dashboard
- Overview, Models, and AI Review sections
- Clear classification and confidence indicators

## Technology Stack

| Component        | Technology            |
| ---------------- | --------------------- |
| Framework        | Next.js 15            |
| Language         | TypeScript            |
| UI Library       | React 19              |
| Styling          | Tailwind CSS 4        |
| Backend API      | FastAPI               |
| Machine Learning | PyTorch, Scikit-learn |
| Frontend Hosting | Vercel                |
| Backend Hosting  | Render                |

The frontend was originally initialized using Create T3 App.

## Application Architecture

```text
                       User
                        |
                        v
                Next.js Frontend
                        |
                 ECG File Upload
                        |
                        v
                  FastAPI Backend
                        |
                 ECG Preprocessing
                        |
             +----------+----------+
             |          |          |
             v          v          v
          Random      CNN +     CTRhythm
          Forest   Transformer
             |          |          |
             +----------+----------+
                        |
                        v
               Prediction Results
                        |
                        v
               AI-Assisted Review
                        |
                        v
                 User Dashboard
```

## Project Structure

```text
web/
├── public/
│
├── src/
│   ├── app/
│   │   ├── page.tsx
│   │   ├── analyze/
│   │   │   └── page.tsx
│   │   ├── results/
│   │   │   └── page.tsx
│   │   ├── api/
│   │   │   └── ecg/
│   │   │       └── predict/
│   │   │           └── route.ts
│   │   └── layout.tsx
│   │
│   ├── styles/
│   ├── types/
│   ├── server/
│   └── trpc/
│
├── package.json
├── next.config.js
├── tsconfig.json
└── README.md
```

## Application Pages

| Route      | Description                                  |
| ---------- | -------------------------------------------- |
| `/`        | Landing page and project introduction        |
| `/analyze` | ECG upload and analysis                      |
| `/results` | Model predictions, confidence, and AI review |

## Getting Started

### Prerequisites

Make sure you have the following installed:

- Node.js
- npm
- Python 3.11 or compatible version
- FastAPI backend dependencies

### 1. Clone the Repository

```bash
git clone https://github.com/tung2307/257-ECG-Demo.git
cd 257-ECG-Demo/web
```

### 2. Install Dependencies

```bash
npm ci
```

### 3. Configure Environment Variables

Create a `.env.local` file inside the `web/` directory:

```env
NEXT_PUBLIC_ECG_API_URL=http://127.0.0.1:8000
ECG_API_URL=http://127.0.0.1:8000
```

`NEXT_PUBLIC_ECG_API_URL` is used for direct requests from the analysis page to the FastAPI backend.

`ECG_API_URL` is used by the Next.js server-side prediction API route.

### 4. Start the Backend

Open a separate terminal from the repository root:

```bash
cd api

python -m pip install -r requirements.txt
python -m pip install torch

uvicorn main:app --reload
```

The backend will be available at:

http://localhost:8000

FastAPI documentation:

http://localhost:8000/docs

Make sure the trained model checkpoints are available in `api/models/`.

### 5. Start the Frontend

From the `web/` directory:

```bash
npm run dev
```

Visit:

http://localhost:3000

## How It Works

### Step 1 — Upload ECG

Users upload an ECG recording using the analysis interface.

Supported formats:

- MATLAB `.mat`
- CSV `.csv`

The application supports analyzing up to 12 recordings in one session.

### Step 2 — ECG Preprocessing

The backend processes uploaded ECG signals before classification.

The deployed inference pipeline uses:

- Target sampling frequency: 150 Hz
- Fixed duration: 60 seconds
- Signal normalization
- Padding or truncation

### Step 3 — Model Inference

Each recording is processed by three machine learning models.

**Random Forest**

A traditional machine learning classifier serving as the baseline.

**CNN + Transformer**

Combines convolutional feature extraction with Transformer-based temporal modeling.

**CTRhythm**

Implements an architecture based on the IEEE BIBM 2024 paper, combining CNN layers and Transformer encoders to capture both local and long-range ECG patterns.

### Step 4 — Review Predictions

The application displays:

- Predicted rhythm category
- Model confidence
- Class probabilities
- Inference runtime
- Model performance information

### Step 5 — AI Review

When the AI service is configured, the application generates an educational explanation of the predictions, model agreement, and potential follow-up considerations.

## Experimental Results

The following metrics were recorded during the project.

| Model             | Accuracy | Macro F1 | AFib Recall | AFib F1 |
| ----------------- | -------: | -------: | ----------: | ------: |
| Random Forest     |   0.5955 |   0.2505 |        0.00 |    0.00 |
| CNN + Transformer |   0.8535 |   0.7963 |        0.76 |    0.78 |
| CTRhythm (Fold 1) |   0.8658 |   0.7709 |         N/A |  0.8121 |

### Key Findings

- CTRhythm achieved the highest recorded accuracy at 86.58%.
- CNN + Transformer achieved the highest four-class Macro F1 at 0.7963.
- CTRhythm achieved the highest recorded AFib F1 at 0.8121.
- Random Forest performed substantially below the deep learning models.

The CTRhythm checkpoint achieved a paper-style F1 score of 0.8427, calculated across Normal, AFib, and Other.

These are saved project validation results. CTRhythm represents a selected Fold 1 checkpoint rather than a full five-fold average.

For more information, see [Experimental Results](../docs/RESULTS.md).

## Testing

### TypeScript Validation

```bash
npm run typecheck
```

### Production Build

```bash
npm run build
```

### Format Checking

```bash
npm run format:check
```

Python evaluation tests are available in the repository's root-level `tests/` directory.

## Deployment

The application uses a separated deployment architecture.

### Frontend — Vercel

The Next.js frontend is deployed on Vercel.

Production URL:

https://257-ecg-demo.vercel.app/

Configure the environment variables in Vercel using the public HTTPS backend URL.

### Backend — Render

The Python FastAPI backend is deployed on Render.

The backend handles:

- ECG signal preprocessing
- Trained model inference
- Prediction aggregation
- AI-assisted summaries

The frontend communicates with the backend using HTTP requests.

Ensure the backend's CORS configuration includes the frontend deployment domain.

## Research Reference

Liang, Z., et al. (2024).

_CTRhythm: Accurate Atrial Fibrillation Detection from Single-Lead ECG by Convolutional Neural Network and Transformer Integration._

2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM).

**DOI:** https://doi.org/10.1109/BIBM62325.2024.10822563

**Official Implementation:** https://github.com/labxscut/CTRhythm

## Limitations

- The system is intended for educational and research use.
- Model predictions depend on the quality of uploaded ECG recordings.
- Model confidence is not a clinically calibrated probability.
- Predictions may differ between models.
- The application is not clinically validated.
- The original training notebooks and split metadata are required for complete reproduction of historical checkpoint results.

## Contributors

**San José State University — CMPE 257**

- Tung Nguyen
- Angie Do
- Neel Patel

## Disclaimer

This application is intended solely for educational and machine learning research purposes.

It is not a certified medical device and must not be used for medical diagnosis, treatment decisions, or replacement of professional medical advice.
