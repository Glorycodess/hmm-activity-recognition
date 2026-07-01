# HMM-Based Human Activity Recognition

A Hidden Markov Model (HMM) built from scratch to classify human physical activities — Still, Standing, Walking, and Jumping — from smartphone accelerometer and gyroscope data.

## Project Overview

This project implements a complete activity recognition pipeline using real sensor data collected via the Sensor Logger app on an iPhone. The HMM is implemented from scratch in Python (NumPy), including the Viterbi decoding algorithm and Baum-Welch Expectation-Maximization training with a log-likelihood convergence check. Results are validated against the hmmlearn library.

**Training accuracy:** 96.2% | **Test accuracy (unseen data):** 90.2%

## Repository Structure

```
hmm-activity-recognition/
├── data/                  # Labeled sensor recordings (CSV)
│   ├── still/             # 15 training takes
│   ├── standing/          # 15 training takes
│   ├── walking/           # 15 training takes
│   ├── jumping/           # 15 training takes
│   └── test/              # 4 unseen test recordings (1 per activity)
├── outputs/               # Generated plots and visualizations
├── report/                # Final report (PDF)
├── Formative2.ipynb       # Main notebook — full pipeline
└── README.md
```

## Pipeline

1. Load and merge accelerometer + gyroscope readings per take
2. Segment into 1-second windows (50 samples at 50 Hz) and extract 16 features (14 time-domain, 2 frequency-domain via FFT)
3. Z-score normalize features
4. Initialize HMM parameters from labeled data (supervised initialization)
5. Refine with Baum-Welch EM (converges in 7 iterations)
6. Decode activity sequences using Viterbi algorithm
7. Evaluate on 4 unseen test recordings across all activities

## Data Collection

- Device: iPhone (Sensor Logger app by Chio Tsz Hei)
- Sensors: Accelerometer (x, y, z) and Gyroscope (x, y, z)
- Sampling rate: 50 Hz
- Activities: Still, Standing, Walking, Jumping
- Training data: 15 takes × 4 activities × ~10 seconds each (~150s per activity)
- Test data: 1 unseen take per activity, collected in a separate session

## Results

| Activity | Sensitivity | Specificity | Accuracy |
|---|---|---|---|
| Still | 1.000 | 1.000 | 1.000 |
| Standing | 0.667 | 1.000 | 0.918 |
| Walking | 0.933 | 0.913 | 0.918 |
| Jumping | 1.000 | 0.956 | 0.967 |

## Requirements

numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, hmmlearn

## Author

Glory Paul — African Leadership University, June 2026
Formative 2 | Machine Learning Operations
