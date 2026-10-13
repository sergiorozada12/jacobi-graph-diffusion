# DiPhon: Diffusion on Graphons for Scalable Graph Generation

Official implementation of **DiPhon: Diffusion on Graphons for Scalable Graph Generation**. The repository provides a unified command-line interface for training graph diffusion models, generating and evaluating graphs, and tuning sampler hyperparameters.

## Status and scope

The supported configurations are:

| CLI model | Dataset file | Graph type | Training mode | Default checkpoint directory |
| --- | --- | --- | --- | --- |
| `pa` | `data/pa_graphon.pkl` | Preferential attachment | Binary graph | `checkpoints/pa_graphon/` |
| `sbm_2comms` | `data/sbm_2comms_graphon.pkl` | Two-community SBM | Binary graph | `checkpoints/sbm_2comms_graphon/` |
| `tree_graphon` | `data/tree_graphon.pkl` | Trees | Binary graph | `checkpoints/tree_graphon/` |
| `d_regular` | `data/d_regular.pkl` | 4-regular graphs | Binary graph | `checkpoints/d_regular/` |
| `metrofi_cond` | `data/metrofi/metrofi.pkl` | MetroFi weighted graphs, location-conditioned | Weighted graph | `checkpoints/metrofi-cond-nosf/` |
| `metrofi_uncond` | `data/metrofi/metrofi.pkl` | MetroFi weighted graphs, unconditional | Weighted graph | `checkpoints/metrofi-uncond-pe-nosf/` |

The `data/`, `checkpoints/`, `results/`, `samples/`, and `hyperparams/` directories are ignored by Git. A fresh clone therefore does **not** contain datasets or pretrained weights. Generate the datasets as described below and download the pretrained artifacts when needed.

## Pretrained artifacts and generated samples

- [Paper checkpoints (Google Drive)](https://drive.google.com/drive/folders/1iTHROOLrLeHRl74TfTaPzmbCQFLI0MQa?usp=sharing): pretrained checkpoints for the results reported in the paper. Download them and preserve the model-specific directory structure under `checkpoints/`, or pass a downloaded weight file directly with `--checkpoint PATH`.
- [Generated graph examples (Google Drive)](https://drive.google.com/drive/folders/1lrBzQrYYI6aXGYe58ADLrQ4QEJ8zfJD3?usp=sharing): example graphs sampled from the trained models. Synthetic graph pickles can be evaluated without resampling by passing the downloaded file with `--load-graphs-path PATH`.

These files are hosted externally and are not downloaded automatically by the repository.

## Repository layout

```text
configs/                    Model, data, SDE, sampler, and tuning settings
src/dataset/                PyTorch Lightning data modules
src/dataset_generation/     Synthetic and MetroFi dataset builders
src/features/               Structural feature extraction
src/metrics/                Graph and MetroFi evaluation metrics
src/models/                 Graph transformer architecture
src/parameter_search/       Grid search and persistent trial storage
src/sample/                 Reverse-SDE solvers and graph sampler
src/sde/                    Jacobi SDE and score functions
src/train/                  Lightning training modules
src/visualization/          Plotting utilities
main.py                     Unified train, generation/test, and tuning CLI
environment.yml             Reproducible Conda environment
```

## Installation

The supplied environment targets Linux and Python 3.9.

```bash
git clone https://github.com/sergiorozada12/jacobi-graph-diffusion.git
cd jacobi-graph-diffusion
conda env create -f environment.yml
conda activate graphon
```

Compile the ORCA executable used by orbit metrics:

```bash
g++ -O2 -std=c++11 -o src/metrics/orca/orca src/metrics/orca/orca.cpp
```

On systems where importing `graph_tool` and PyTorch produces an OpenMP symbol error such as `GOMP_5.0 not found`, preload the system OpenMP library before running the project:

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1
```

Confirm that the CLI imports correctly:

```bash
python main.py --help
```

## Datasets

### Required format

The synthetic data loader expects a pickle containing three lists of NetworkX graphs:

```python
{
    "train": [...],
    "val": [...],
    "test": [...],
}
```

The filenames must match the table in [Status and scope](#status-and-scope), unless the corresponding configuration is changed.

### Synthetic datasets

Generate all supported synthetic datasets with:

```bash
python src/dataset_generation/gen_synth.py
```

This writes the PA, tree, two-community SBM, and 4-regular training datasets:

```text
data/pa_graphon.pkl
data/tree_graphon.pkl
data/sbm_2comms_graphon.pkl
data/d_regular.pkl
```

The same command also creates the fixed-size 4-regular evaluation datasets `data/d_regular_<size>.pkl` for sizes 20 through 200. Use `--output-dir PATH` to write all generated datasets elsewhere.

The generator parameters and random seeds are defined at the top of `gen_synth.py`. Dataset generation overwrites files with matching names.

### MetroFi

Place the raw whitespace-delimited MetroFi measurements at:

```text
data/metrofi/stumble_filtered.txt
```

Each row must contain `MAC address`, `latitude`, `longitude`, and raw RSSI. Build the graph dataset with:

```bash
python -m src.dataset_generation.gen_metrofi_dataset
```

The command writes `data/metrofi/metrofi.pkl` and diagnostic figures. Existing output is protected by default; use `--force` to replace it. Run the following for all builder options:

```bash
python -m src.dataset_generation.gen_metrofi_dataset --help
```

MetroFi graphs use a shared access-point node universe and weighted edges. The conditional model additionally consumes normalized location coordinates.

## Command-line interface

The entry point has three workflows:

```text
python main.py train ...   Train a model
python main.py gen ...     Generate and evaluate graphs
python main.py test ...    Alias for gen
python main.py tune ...    Search sampler/SDE hyperparameters
```

Common arguments:

| Argument | Meaning |
| --- | --- |
| `--model MODEL` | One of the supported model names listed above |
| `--device DEVICE` | `cpu`, `mps`, `cuda`, or a specific device such as `cuda:1` |
| `--seed N` | Override the configuration seed |
| `--batch-size N` | Override the data-loader batch size |
| `--wandb` / `--no-wandb` | Enable or disable Weights & Biases logging |

Use the subcommand help as the authoritative list of options:

```bash
python main.py train --help
python main.py gen --help
python main.py tune --help
```

## Training

Examples:

```bash
# Synthetic graph model
python main.py train --model tree_graphon --device cuda:0

# Conditional weighted MetroFi model without W&B
python main.py train --model metrofi_cond --device cuda:0 --no-wandb
```

Training behavior:

- Hyperparameters come from `configs/config_<model>.py`.
- Checkpoints are written under the directory shown in the model table.
- `last.ckpt` is resumed automatically when present.
- Epoch checkpoints are saved at the interval configured by `save_checkpoint_every_n_epochs`.
- Training exports `weights.pth` and, when EMA is enabled, `weights_ema.pth` for generation.
- `--device cuda:N` selects that GPU; `--device cuda` lets Lightning select the available GPU devices.

To restart training from scratch, move or remove the corresponding `last.ckpt` first. Preserve it elsewhere if the run may be needed later.

## Generation and evaluation

`gen` and `test` are equivalent. When no checkpoint is supplied, the CLI searches the model's checkpoint directory for `weights_ema.pth` and then `weights.pth`. Pass `--no-ema` to exclude EMA weights.

### Synthetic graph example

```bash
python main.py test \
  --model sbm_2comms \
  --device cuda:0 \
  --num-samples 500 \
  --sampling-batch-size 50 \
  --min-nodes 100 \
  --max-nodes 100 \
  --checkpoint checkpoints/sbm_2comms_graphon/weights_ema.pth \
  --save-graphs-path samples/sbm_n100.pkl \
  --json-out results/sbm_n100.json
```

For synthetic models, evaluation reports distributional graph metrics and dataset-specific validity/accuracy metrics. When `--min-nodes` and `--max-nodes` specify the same size, the CLI also attempts to load size-specific references. Disable that comparison with `--skip-size-ref`.

Useful evaluation arguments:

| Argument | Meaning |
| --- | --- |
| `--checkpoint PATH` | Explicit `.ckpt` or `.pth` weights |
| `--load-graphs-path PATH` | Evaluate an existing graph pickle instead of sampling; synthetic models only |
| `--num-samples N` | Number of generated graphs |
| `--sampling-batch-size N` | Sampling batch size, independent of the data-loader batch size |
| `--min-nodes N --max-nodes N` | Restrict sampled graph sizes; both must be supplied |
| `--n-folds N` | Split generated samples into evaluation folds |
| `--expected-num-graphs N` | Fail unless exactly this many graphs are available |
| `--save-graphs-path PATH` | Save generated NetworkX graphs |
| `--json-out PATH` | Save machine-readable metrics |
| `--no-ema` | Do not load EMA weights |
| `--skip-size-ref` | Skip size-specific reference metrics |

Sampler and SDE settings can be overridden from the CLI. See `python main.py gen --help` for the complete set, including predictor, corrector, SNR, solver steps, time schedule, and epsilon options.

### MetroFi evaluation

Run full conditional evaluation with:

```bash
python main.py test \
  --model metrofi_cond \
  --device cuda:0 \
  --checkpoint checkpoints/metrofi-cond-nosf/weights_ema.pth \
  --num-samples 128 \
  --wireless-output-dir samples/metrofi_cond \
  --json-out results/metrofi_cond.json
```

Run only the paired masked evaluation:

```bash
python main.py test \
  --model metrofi_cond \
  --checkpoint checkpoints/metrofi-cond-nosf/weights_ema.pth \
  --conditional-eval-only \
  --conditional-eval-split test \
  --conditional-eval-graphs 943 \
  --conditional-eval-condition-mode true \
  --conditional-eval-mask-mode both \
  --json-out results/metrofi_cond_masked.json
```

MetroFi-specific evaluation can compare true, shuffled, or zeroed coordinate conditions and true, full, or both observation masks. Long masked evaluations support `--conditional-eval-resume-path`.

## Hyperparameter tuning

Search spaces and objectives are defined in `configs/config_tune.py`. For example:

```bash
python main.py tune \
  --model tree_graphon \
  --device cuda:0 \
  --checkpoint checkpoints/tree_graphon/weights_ema.pth \
  --num-samples 100 \
  --sampling-batch-size 25 \
  --min-nodes 80 \
  --max-nodes 80 \
  --store-name tree_n80
```

Trial state is persisted in `hyperparams/<store-name>.json`. Use a store identifier without the `.json` extension. Successful parameter combinations are skipped on later runs using the same store, allowing interrupted searches to continue.

The configured objectives are `pa_acc`, `sbm_acc`, `tree_acc`, `4_regular_acc`, and `average_ratio` for MetroFi. Edit `get_tune_config()` in `configs/config_tune.py` to change a search grid.

## Outputs

| Path | Contents |
| --- | --- |
| `checkpoints/<experiment>/` | Lightning checkpoints and exported model/EMA weights |
| `samples/` | Generated graph pickles and diagnostic plots by default |
| `results/` | JSON metrics and MetroFi masked-evaluation artifacts |
| `hyperparams/` | Persistent tuning histories |
| `data/ref_metrics/` | Optional cached or size-specific synthetic reference metrics |

Generation logs print the effective model, dataset, device, seed, sample count, checkpoint path, saved artifacts, metrics, warnings, and final status. Fatal evaluation errors propagate with a nonzero exit code.

## Reproducibility notes

- Each model configuration defines its default seed.
- CLI `--seed` overrides that value and seeds PyTorch Lightning workers.
- Dataset-generation scripts use fixed NumPy/NetworkX seeds.
- Exact numerical reproducibility across CUDA, PyTorch, and `graph_tool` versions is not guaranteed.
- Record the command, commit hash, dataset checksum, checkpoint, and generated JSON for archival experiments.

## Troubleshooting

**Dataset not found**

Check that the expected pickle from the model table exists. Datasets are not downloaded automatically.

**Checkpoint not found**

Supply `--checkpoint PATH` or place `weights_ema.pth`/`weights.pth` in the expected checkpoint directory.

**ORCA errors**

Recompile `src/metrics/orca/orca` and ensure it is executable:

```bash
chmod +x src/metrics/orca/orca
```

**CUDA out of memory**

Reduce `--sampling-batch-size` for generation or `--batch-size` for training. These controls are independent.

**W&B authentication or connectivity errors**

Use `--no-wandb` for an offline training run.
