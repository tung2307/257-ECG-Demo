# Experimental Results

## Model Performance Comparison

Three ECG classification models were evaluated during the project.

| Model             | Accuracy | Macro F1 |   AFib Recall | AFib F1 |
| ----------------- | -------: | -------: | ------------: | ------: |
| Random Forest     |   0.5955 |   0.2505 |          0.00 |    0.00 |
| CNN + Transformer |   0.8535 |   0.7963 |          0.76 |    0.78 |
| CTRhythm (Fold 1) |   0.8658 |   0.7709 | Not available |  0.8121 |

These values are from previously saved project results, not a new evaluation run.

## Paper-Style F1 Comparison

The CTRhythm paper computes an overall F1 score by averaging the F1 scores for Normal, AFib, and Other, excluding Noisy.

| Model                                   | Paper-Style F1 |
| --------------------------------------- | -------------: |
| Original CTRhythm (without pretraining) |         0.8310 |
| Original CTRhythm (with pretraining)    |         0.8400 |
| Our CNN + Transformer                   |         0.8230 |
| Our CTRhythm (Fold 1)                   |         0.8427 |

The original paper reports five-fold cross-validation averages.

Our deployed CTRhythm model uses a selected Fold 1 validation checkpoint. Therefore, these results are not directly equivalent.

## Key Findings

### 1. Random Forest Baseline

Random Forest achieved 59.55% accuracy and 0.2505 macro F1.

The saved experiment reported no successful AFib detection under the evaluated conditions.

### 2. CNN + Transformer

The CNN + Transformer achieved 85.35% accuracy and a four-class macro F1 of 0.7963.

This was the highest four-class macro F1 among our three saved models.

### 3. CTRhythm

CTRhythm achieved 86.58% accuracy and an AFib F1 of 0.8121.

It obtained the highest recorded accuracy and AFib F1 among the three models.

However, its four-class macro F1 was lower than the simplified CNN + Transformer model.

## Evaluation Metrics

- Accuracy: Proportion of correctly classified recordings.
- Macro F1: Unweighted mean of F1 across four classes.
- AFib Recall: Proportion of actual AFib samples correctly identified.
- AFib F1: Harmonic mean of AFib precision and recall.
- Paper-Style F1: Mean F1 for Normal, AFib, and Other.

## Limitations

The exact CTRhythm Fold 1 AFib recall was not preserved in the available project summary.

The original validation predictions or confusion matrix are required to recover this value.

The reference paper and our Fold 1 experiment use different evaluation aggregations.

Further reproducibility work should include:

- Original training code
- Dataset split manifests
- Random seeds and hyperparameters
- Per-recording validation predictions
- Training and validation loss curves
- Confusion matrices
