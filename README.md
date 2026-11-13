# ECG-Based Atrial Fibrillation Detection

**CMPE 257 — Machine Learning | San José State University**

An end-to-end machine learning project for classifying single-lead electrocardiogram (ECG) recordings using three models:

- Random Forest (traditional ML baseline)
- CNN + Transformer (deep learning baseline)
- CTRhythm (research paper implementation)

The project evaluates classification performance and provides an interactive web interface for ECG analysis.

**Team Members:** Tung Nguyen, Angie Do, Neel Patel

**Live Demo:** https://257-ecg-demo.vercel.app/

## Research Objective

Atrial Fibrillation (AFib) is a cardiac rhythm abnormality characterized by irregular electrical activity.

Our goal is to compare traditional machine learning and deep learning approaches for classifying ECG recordings into four categories:

| Class  | Description                    |
| ------ | ------------------------------ |
| Normal | Normal heart rhythm            |
| AFib   | Atrial fibrillation            |
| Other  | Other cardiac rhythms          |
| Noisy  | Unclear or noisy ECG recording |

The research investigates whether combining convolutional feature extraction with Transformer-based temporal modeling improves classification performance.

## Reference Paper

Liang et al. (2024).

_CTRhythm: Accurate Atrial Fibrillation Detection from Single-Lead ECG by Convolutional Neural Network and Transformer Integration._

IEEE International Conference on Bioinformatics and Biomedicine (BIBM).

Paper: https://doi.org/10.1109/BIBM62325.2024.10822563

Official Implementation:
https://github.com/labxscut/CTRhythm

## Model Architectures

### Random Forest

A conventional supervised learning baseline using engineered ECG features.

### CNN + Transformer

A hybrid deep learning architecture combining:

1. CNN layers for local signal feature extraction.
2. Transformer layers for temporal relationships.
3. Classification layers for ECG rhythm prediction.

### CTRhythm

The paper-based architecture integrates CNN feature extraction and Transformer encoders to model both local and long-range ECG characteristics.

The deployed CTRhythm checkpoint uses Fold 1 validation results.

## Dataset and Preprocessing

The reference experiment uses the PhysioNet/Computing in Cardiology Challenge 2017 dataset.

Dataset:
https://physionet.org/content/challenge-2017/1.0.0/

The deployed API processes ECG signals using:

- Target sampling frequency: 150 Hz
- Signal duration: 60 seconds
- Signal normalization
- Padding/truncation for fixed input length

The training and validation procedures are documented in `docs/EXPERIMENTS.md`.

## Experimental Results

| Model             | Accuracy | Macro F1 | AFib Recall | AFib F1 |
| ----------------- | -------: | -------: | ----------: | ------: |
| Random Forest     |   0.5955 |   0.2505 |        0.00 |    0.00 |
| CNN + Transformer |   0.8535 |   0.7963 |        0.76 |    0.78 |
| CTRhythm (Fold 1) |   0.8658 |   0.7709 |         N/A |  0.8121 |

Macro F1 represents the unweighted average across all four classes.

The reported CTRhythm Fold 1 paper-style F1 score is 0.8427, excluding the Noisy class.

See `docs/RESULTS.md` for additional interpretation and limitations.

## Technology Stack

| Component           | Technology                        |
| ------------------- | --------------------------------- |
| Machine Learning    | PyTorch, Scikit-learn             |
| Backend             | Python, FastAPI                   |
| Frontend            | Next.js, TypeScript, Tailwind CSS |
| Frontend Deployment | Vercel                            |
| Backend Deployment  | Render                            |

## Repository Structure

```text
257-ECG-Demo/
├── api/                  # FastAPI and trained models
├── web/                  # Next.js application
├── configs/              # Experiment settings
├── docs/                 # Research documentation
├── scripts/              # Evaluation utilities
├── tests/                # Automated tests
├── .github/workflows/    # Continuous integration
└── README.md
```

## Running the Application Locally

### Backend

From the repository root:

```bash
cd api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch
uvicorn main:app --reload
```

Backend URL: http://localhost:8000

The saved model checkpoints must be present in `api/models/`.

### Frontend

Open a separate terminal:

```bash
cd web
npm ci
```

Create `web/.env.local`:

```env
ECG_API_URL=http://127.0.0.1:8000
```

Start the frontend:

```bash
npm run dev
```

Open http://localhost:3000

## Reproducing Evaluation Metrics

Install the evaluation dependencies:

```bash
pip install scikit-learn pytest
```

Export held-out predictions with columns:

```text
true_label,pred_label
```

Then run:

```bash
python scripts/evaluate_predictions.py \
  --input results/predictions.csv \
  --output results/evaluation.json
```

The script produces accuracy, macro F1, AFib recall, AFib F1, paper-style F1, and a confusion matrix.

The original training split and code must be recovered to fully reproduce the saved checkpoint results.

## Documentation

- `docs/EXPERIMENTS.md` — Experimental methodology
- `docs/RESULTS.md` — Model comparison and findings
- `configs/experiment_plan.yaml` — Experiment configuration
- `scripts/evaluate_predictions.py` — Evaluation implementation

## Limitations

- Model predictions are for research and educational purposes.
- The system is not validated as a clinical diagnostic device.
- CTRhythm results are from a selected Fold 1 checkpoint, while the reference paper uses five-fold cross-validation.
- The original checkpoint training scripts and split metadata are not yet included in this repository.
- Model confidence is not equivalent to clinical certainty.

## License and Attribution

This project was developed for CMPE 257 at San José State University.

CTRhythm is based on the referenced IEEE BIBM 2024 publication. Original research and implementation remain attributed to their respective authors.
