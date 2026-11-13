# Experiment Methodology

## Research Question

How do traditional machine learning and CNN-Transformer architectures compare in ECG rhythm classification, particularly for atrial fibrillation detection?

## Models

### Experiment 1: Random Forest

Purpose: Establish a traditional machine learning baseline.

Implementation:

- Engineered ECG features
- Scikit-learn Random Forest classifier
- Four-class ECG classification

Saved checkpoint:
`api/models/ecg_random_forest.pkl`

### Experiment 2: CNN + Transformer

Purpose: Evaluate a simplified hybrid deep learning architecture.

Architecture:

- CNN feature extraction
- Transformer temporal modeling
- Classification output

Saved checkpoint:
`api/models/ecg_cnn_transformer_best.pth`

### Experiment 3: CTRhythm

Purpose: Evaluate the CNN-Transformer architecture described in the reference paper.

Architecture:

- Convolutional feature encoder
- Residual feature extraction
- Transformer encoder
- Classification module

Saved checkpoint:
`api/models/ctrhythm_fold1_best.pth`

The deployed model uses a Fold 1 checkpoint.

## Dataset

Reference: PhysioNet/CinC 2017.

The dataset contains 8,528 labeled single-lead ECG recordings.

Classes:

- Normal
- AFib
- Other
- Noisy

## Preprocessing

The deployed backend standardizes ECG signals to:

- Sampling rate: 150 Hz
- Duration: 60 seconds
- Fixed-length input
- Signal normalization

Exact historical training preprocessing should be verified against the original training notebook.

## Evaluation

Metrics:

1. Accuracy
2. Four-class Macro F1
3. AFib Recall
4. AFib F1
5. Paper-Style F1

Paper-Style F1:

F1_paper = (F1_Normal + F1_AFib + F1_Other) / 3

Four-class Macro F1:

F1_macro = (F1_Normal + F1_AFib + F1_Other + F1_Noisy) / 4

## Reproducibility Requirements

The following historical details still need verification:

| Detail                            | Status       |
| --------------------------------- | ------------ |
| Model checkpoints                 | Available    |
| Architecture and inference source | Available    |
| Saved performance metrics         | Available    |
| Original training notebooks       | Not included |
| Random seeds                      | To recover   |
| Exact training epochs             | To recover   |
| Learning rate and optimizer       | To recover   |
| Fold assignments                  | To recover   |
| Validation predictions            | To recover   |

Do not substitute a newly generated split for the original Fold 1 validation split.

## Future Experiments

Planned improvements include:

- Full five-fold cross-validation
- Per-class confusion matrices
- Multiple random seeds
- Training curve visualizations
- Equal-split comparison across all three models
- Inference runtime benchmarking
