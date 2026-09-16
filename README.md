# DSGBG: Dual-Scale Granular Ball Generation

Source code for the paper *"DSGBG: An adaptive granular-ball generation method based on dual-scale cut-off distance and inter-class separation degree perception"*.

## Description

DSGBG is an adaptive granular-ball generation method. It combines the coefficient of variation (cv) with the inter-class separation degree (λ) to construct a dual-scale cut-off distance, and generates pure granular balls via score-based greedy center selection and radius truncation by the nearest heterogeneous distance.

This repository provides the paper implementations of **DSGBG and 6 baseline granular-ball generation methods**, plus three classifiers (GBKNN / GBKNN++ / IGBKNN).

## Algorithms

| # | Algorithm | Function | Description |
|---|-----------|----------|-------------|
| 1 | **DSGBG** | `fit_dsgbg` | Dual-scale cut-off distance with cv-λ fusion, blend labels, outlier re-selection |
| 2 | ScOrGBC | `fit_scorgbg` | K-means++ stable centers and optimal radii |
| 3 | LDGBG | `fit_ldgbg` | Local-density-based granular ball generation |
| 4 | ORIGBG | `fit_origb` | Purity-driven recursive 2-means splitting |
| 5 | ACCGBG | `fit_accgbg` | K-division accelerated adaptive generation |
| 6 | ADPGBG | `fit_adpgbg` | Adaptive generation based on shortest heterogeneous distance |
| 7 | GBG++ | `fit_gbgpp` | Attention-driven fast and stable generation |

## Usage

```python
import numpy as np
from granular_ball_algorithms import fit_dsgbg, gbknn_predict

# X: (N, M) sample matrix, y: (N,) labels
centers, radii, labels, ball_sizes = fit_dsgbg(X, y)
pred = gbknn_predict(X_test, centers, radii, labels, ball_sizes)
```

Classifier assignment by algorithm (same as the paper experiments):

| Algorithm | Classifier |
|-----------|------------|
| DSGBG, ScOrGBC, LDGBG, ORIGBG, ACCGBG | GBKNN |
| GBG++ | GBKNN++ |
| ADPGBG | IGBKNN |

## Requirements

```bash
pip install -r requirements.txt
```

- numpy
- scikit-learn
- pandas

## Datasets

All experiments use 25 benchmark datasets collected from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/) and [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/):
Iris, Parkinsons, Seeds, Votes, Haberman, Ionosphere, Dermatology6, Chscase-vine2, WDBC, Breastcancer, Austra, Diabetes, Vehicle3, Fourclass, Credit-g, Yeast, Segment0, Image_segmentation, Svmguide1, Waveform, Mushroom, Pen, Credit, Adult, Firewall.

**Preprocessing:** Min-max normalization; 5-times 5-fold stratified cross-validation. In noise experiments, 10%/20%/30%/40% label noise is injected into training labels (flipped equiprobably to other classes); test labels remain clean. Parameters are fixed at their clean-label optimal values across all noise levels.


