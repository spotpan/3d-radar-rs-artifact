# Physics-guided Randomized Smoothing for 3D Radar QPE

This repository provides the implementation of a physics-guided randomized smoothing framework for uncertainty diagnosis in 3D radar quantitative precipitation estimation (QPE).

The code supports:

- 3D masked-autoencoder based radar representation learning.
- Multi-frame 3D radar precipitation retrieval.
- Patch-level precipitation-event evaluation.
- Physics-guided randomized smoothing under radar-specific perturbations.
- Uncertainty analysis for temporal-frame dropout, vertical-level dropout, vertical-level scaling, local block masking, and mixed structured perturbations.
- Height-layer sensitivity analysis for low-, middle-, and high-level radar slices.

## Repository Structure

```text
.
├── configs/              # Configuration files
├── data/                 # Dataset and dataloader definitions
├── evaluation/           # Evaluation and randomized smoothing scripts
├── models/               # 3DMAE encoder and QPE decoder models
├── scripts/              # Running scripts and utility scripts
├── training/             # Pretraining and fine-tuning scripts
├── utils/                # Utility functions and metrics
├── requirements.txt      # Python dependencies
└── README.md


