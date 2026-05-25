# DiPhon: Diffusion on Graphons for Scalable Graph Generation

This repository contains the official implementation of **DiPhon: Diffusion on Graphons for Scalable Graph Generation**, a framework for generating graphs via continuous graph-diffusion processes.

---

## 🚀 Architectural Overview

To ensure modularity and convenience, all independent execution pipelines (training, evaluation, generation, and grid-search tuning) have been consolidated into a single, cohesive command-line interface:

```
                      ┌───────────────────────┐
                      │    main.py (CLI)      │
                      └───────────┬───────────┘
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
 ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
 │    train     │         │     gen      │         │     tune     │
 └──────┬───────┘         └──────┬───────┘         └──────┬───────┘
        ▼                        ▼                        ▼
PyTorch Lightning       Graph Sampling, EMD     Grid hyperparameter
  Training Loop         & Size-Matched Ratios   tuning via config_tune
```

### Key Components:
1. **Unified CLI Entrypoint (`main.py`)**: A single orchestrator managing all execution flows via `argparse` subparsers (`train`, `gen`, and `tune`).
2. **Centralized Configs (`configs/`)**: Model, dataset, and sampler hyperparameter configurations are managed inside dedicated python configuration classes (e.g. `config_pa.py`, `config_sbm.py`, etc.).
3. **Centralized Tuning Grids (`configs/config_tune.py`)**: All grid hyperparameter search ranges, target objectives, and validation metrics aliases are managed in one centralized search grid.

---

## 📦 Installation & Setup

Setting up the project requires configuring the conda environment and building the C++ Orca analyzer:

### 1. Create Conda Environment
Initialize the environment containing all Python dependencies:
```bash
conda env create -f environment.yml
conda activate graphon
```

### 2. Compile the Orca C++ Analyzer
Orca is a C++ tool used for fast graph orbit and motif count evaluations. You must compile the C++ binary manually:
```bash
cd src/metrics/orca
g++ -O2 -std=c++11 -o orca orca.cpp
cd ../../..
```
*Note: The generated executable `orca` must remain in `src/metrics/orca/` so that the evaluation scripts can call it successfully.*

### 3. Linux Preloading Workaround
Importing `graph_tool` alongside PyTorch on Linux systems may trigger OpenMP conflict crashes (such as `GOMP_5.0 not found`). Preload your system's OpenMP library when running commands:
```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1
```

---

## 🛠️ CLI Reference & subcommands

`main.py` exposes common flags that apply to all subcommands:
*   `--model`: The target graph diffusion config (`pa`, `sbm`, `sbm_2comms`, `tree`, `tree_graphon`, `planar`, `metrofi`).
*   `--seed`: Override the random seeding dynamically.
*   `--device`: Override target execution device (e.g. `cuda:0`, `cpu`).

### 1. Training Mode (`train`)
Starts the PyTorch Lightning model fitting flow for any dataset. Checkpoint files are stored automatically under `checkpoints/{dataset_name}/`.
```bash
python main.py train --model pa --device cuda:0
```

---

### 2. Generation & Evaluation Mode (`gen`)
Loads pre-trained checkpoints, generates synthetic structures, and evaluates structural metric ratios against reference datasets.
```bash
python main.py gen --model pa --num-samples 500 --min-nodes 80 --max-nodes 80 --checkpoint checkpoints/pa_graphon/epoch=16999.ckpt
```

#### Key Parameters:
*   `--checkpoint`: Path to custom model weights (fully compatible with `.ckpt` or `.pth`).
*   `--num-samples`: Override the target number of synthetic graphs to generate.
*   `--min-nodes` / `--max-nodes`: Enforce strict node size-matched generation ranges.
*   `--save-graphs-path`: Save generated graphs to a custom pickle path.
*   `--load-graphs-path`: Skip generator sampling and instantly evaluate pre-generated graphs.
*   `--json-out`: Write final structural comparison ratios to a standardized JSON dataset.

#### Terminal UI
The generation subparser prints metric comparisons in polished console formats, complete with descriptive status levels:
*   🟢 `[SUCCESS]`: Successful loading/saving operations.
*   🔹 `[INFO]`: System status notifications.
*   **Alighted Data Columns**: Aligned metric evaluation statistics and reference ratios.

---

### 3. Hyperparameter Tuning Mode (`tune`)
Launches sequential grid tuning searches using search parameters retrieved from `configs/config_tune.py`.
```bash
python main.py tune --model pa --min-nodes 80 --max-nodes 80 --store-name pa_tuning_results.json
```

#### Centralized Search Space Configuration:
To update search spaces or target optimization goals (e.g. `pa_acc`, `sbm_acc`, `average_ratio`), edit `configs/config_tune.py` directly:
```python
# configs/config_tune.py
GridSearchSpaces = {
    "pa": SearchSpace(
        n_steps=[50, 100],
        time_schedule=["milstein"],
        # Add hyperparameter grids here
    )
}
```
