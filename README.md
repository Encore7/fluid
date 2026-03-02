# FLUID: Federated Learning with Unlearning and Instant Drift-recovery

Official implementation repository for concept drift adaptation experiments in Federated Learning (FL), including:

- `fedavg`
- `retraining`
- `rapid-retraining`
- `fedau`
- `fluid`

## Paper

- IEEE Xplore: https://ieeexplore.ieee.org/document/11336437

> Note: The published paper covers results up to **FLUID**.  
> The repository also contains an ongoing **Adaptive-FLUID** extension, which is **not** part of the original publication.

## Overview

Federated Learning (FL) enables collaborative model training without sharing raw client data, but performance degrades under concept drift.  
This repository benchmarks classical and unlearning-based recovery approaches and includes **FLUID**, which combines:

- **FedAU** for lightweight auxiliary unlearning of obsolete patterns
- **Rapid Retraining (RRT)** for fast drift recovery

without requiring full retraining from scratch.

## Abstract (Paper Summary)

FLUID frames concept drift adaptation in federated learning as a selective forgetting problem.  
Instead of full retraining, it combines federated unlearning (FedAU) with rapid retraining dynamics to remove obsolete knowledge while learning new patterns.  
In reported experiments, FLUID provides balanced behavior across drift mitigation, recovery speed, and convergence compared with FedAvg, retraining, and FedAU-style baselines.

## Implemented Methods

| Method | Description |
|---|---|
| `fedavg` | Standard federated averaging baseline with natural recovery only |
| `retraining` | Full model reinitialization and retraining after drift |
| `rapid-retraining` | Fast partial retraining for quicker recovery |
| `fedau` | Federated Auxiliary Unlearning under drift |
| `fluid` | Proposed method combining unlearning + rapid recovery |

## Repository Structure

```text
.
├── src/
│   ├── client_app.py          # Flower client logic
│   ├── server_app.py          # Flower server + strategy logic
│   ├── data_loader.py         # Partitioning, drift injection, loaders
│   ├── ml_models/             # CNN model + train/test utilities
│   ├── scripts/               # Dataset preparation helpers
│   └── utils/logger.py        # Per-client logging
├── pre_flwr_run.py            # Clear logs, (optionally) prepare data, then run Flower
├── pyproject.toml             # Main experiment configuration
├── requirements.txt
└── LICENSE
```

## Environment Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Running Experiments

### Recommended: one-command pre-run + execution

```bash
python pre_flwr_run.py
```

This script:
1. Clears logs
2. Prepares dataset partitions (if enabled in config)
3. Launches Flower simulation

### Alternative manual run

```bash
flwr run .
```

Use manual run when data is already prepared and you only want to rerun with a changed strategy/config.

## Configuration (`pyproject.toml`)

Main experiment settings live under:

`[tool.flwr.app.config]`

Most important keys:

- `mode`: one of `retraining`, `rapid-retraining`, `fedavg`, `fedau`, `fluid`
- `prepare-dataset`: set `true` to regenerate client partitions before run
- `num-of-clients`, `num-server-rounds`, `local-epochs`, `batch-size`, `learning-rate`
- `drift-start-round`, `drift-end-round`
- `drift-clients`: JSON-like list as string, e.g. `"[0,1,3]"`
- `incremental-drift-rounds`: JSON map string
- `abrupt-drift-labels-swap`: label swap rules for drift injection
- `dataset-name`: `mnist`, `fashion_mnist`, or `cifar10`
- `dataset-folder-path`, `logs-folder-path`, `plots-folder-path`

## Drift Simulation Concepts

### 1) Label Swaps (What Drift Means Here)

In this repository, concept drift is simulated by **swapping class labels** for selected clients during selected rounds.

Example (MNIST):
- swap `1 <-> 2`
- swap `5 <-> 7`

Interpretation:
- If a training sample originally has label `1`, after swap it is treated as `2`
- If it has label `2`, it is treated as `1`
- Same logic for `5` and `7`

This creates controlled distribution shift without changing the raw images, and allows direct comparison of recovery behavior across methods.

### 2) Incremental Drift (How Drift Grows Over Time)

Instead of applying full drift at once, the repository supports **incremental drift**, where the percentage of affected samples increases over rounds for drifted clients.

Typical schedule (paper-style example):
- Drifted clients: `{1, 5, 7}`
- Drift interval: rounds `20` to `79`
- Round `20-39`: 20% label swap
- Round `40-59`: 40% label swap
- Round `60-79`: 60% label swap
- Round `>=80`: restore to pre-drift labeling (0% swap)

This models realistic gradual behavior change and later return to normal behavior.

### 3) Config Example (`pyproject.toml`)

```toml
drift-start-round = 20
drift-end-round = 80
drift-clients = "[1,5,7]"
abrupt-drift-labels-swap = '[{"label1": 1, "label2": 2}, {"label1": 5, "label2": 7}]'
incremental-drift-rounds = '{"20": 0.2, "40": 0.4, "60": 0.6}'
```

Notes:
- `drift-start-round` is inclusive and `drift-end-round` is exclusive in the code logic.
- With the above config, swapping is active for rounds `20` to `79`, then disabled from round `80`.

### 4) Why This Is Useful

- `FedAvg` shows natural recovery with no dedicated drift handling.
- `Retraining` and `rapid-retraining` show recovery via relearning.
- `FedAU` and `FLUID` show unlearning-driven adaptation.
- Using the same drift schedule makes comparisons fair and reproducible.

## Typical Workflow

1. Set `mode` in `pyproject.toml`.
2. Set `prepare-dataset = true` if dataset partitions need regeneration.
3. Run:
   ```bash
   python pre_flwr_run.py
   ```
4. Check outputs:
   - logs: `log/`
   - prepared datasets: `src/clients_dataset/`
   - metadata: `metadata.json`

## Reproducibility Notes

- Fix and document `num-of-clients`, rounds, drift schedule, and label-swap rules for each run.
- Run multiple seeds/configs for robust method comparison.
- Keep one config snapshot per experiment for supervisor/review reproducibility.

## Citation

If you use this repository, please cite the FLUID paper from IEEE Xplore:

- https://ieeexplore.ieee.org/document/11336437
