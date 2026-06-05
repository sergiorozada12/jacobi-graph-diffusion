# MetroFi Wireless Implementation Notes

This repository includes a MetroFi wireless application path under the `metrofi`
model name. It is implemented as weighted graph generation over a fixed universe
of wireless access points.

## Where The Code Lives

- CLI entry point: `main.py`
- MetroFi config: `configs/config_metrofi.py`
- Dataset loader: `src/dataset/wireless.py`
- Dataset builder: `src/dataset_generation/gen_metrofi_dataset.py`
- Weighted training module: `src/train/trainer_graph.py`
- Shared Lightning training wrapper: `src/train/base_module.py`
- Sampler and reverse solver: `src/sample/sampler.py`, `src/sample/solver.py`
- Wireless metrics: `src/metrics/val.py`, class `WirelessSamplingMetrics`

The code is present, but this checkout does not currently contain the MetroFi
data directory. The configured dataset path is `data/metrofi/metrofi.pkl`, and
the dataset generation script expects raw input at
`data/metrofi/stumble_filtered.txt` by default.

## What The Wireless Setting Means

The MetroFi path treats each location as one graph.

- Nodes are globally observed MAC addresses / access points.
- Every graph uses the same node universe, normally 70 APs from metadata.
- A node has an `observed` attribute when that AP was seen at that location.
- Edges are added between observed AP pairs.
- Edge weights represent inferred interference.
- Training uses continuous edge weights in `[0, 1]`, not binary edges.

The raw dataset builder reads rows with:

```text
mac lat lon rssi
```

It factorizes MAC addresses into node ids and `(lat, lon)` pairs into location
ids. For each location, it computes mean RSSI per observed AP, converts RSSI to
dBm, then to Watts, and creates pairwise interference edges between observed APs.
The supported pairwise interference models are:

- `min`: `min(power_a, power_b)`, the default
- `sum`: `power_a + power_b`
- `product`: `power_a * power_b`

The stored edge value can be dBm or Watts. The default is dBm.

## Dataset Creation

Expected raw input:

```bash
data/metrofi/stumble_filtered.txt
```

Build the processed pickle:

```bash
python src/dataset_generation/gen_metrofi_dataset.py --force
```

Useful options:

```bash
python src/dataset_generation/gen_metrofi_dataset.py \
  --input data/metrofi/stumble_filtered.txt \
  --output data/metrofi/metrofi.pkl \
  --model min \
  --interference-output dbm \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 17 \
  --force
```

For a quick smoke dataset:

```bash
python src/dataset_generation/gen_metrofi_dataset.py \
  --max-locations 100 \
  --force
```

The resulting pickle contains:

- `train`: list of NetworkX graphs
- `val`: list of NetworkX graphs
- `test`: list of NetworkX graphs
- `metadata`: MAC address list, location coordinates, interference metadata

The builder also writes inspection figures under `data/metrofi/figures/`.

## How Loading Works

`WirelessDatasetModule` loads `data/metrofi/metrofi.pkl`, normalizes all edge
weights globally to `[0, 1]`, and preserves the original physical value in
`interference_raw`.

It filters out graphs with too few observed nodes. The default threshold is:

```python
cfg.data.min_observed_nodes = 3
```

Training batches contain:

```python
(node_features, adjacency, observed_mask)
```

The `observed_mask` is important: it tells the model which APs were observed in
that location. The adjacency is padded to the global AP count.

## Training

Train MetroFi with:

```bash
python main.py train --model metrofi --device cuda:0
```

python csub.py -n train-metrofi-bs128 --train --node_type h200 -g 0.3 --command "
conda activate graphon;
cd /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion;
python main.py train --model metrofi --device cuda:0


For CPU debugging:

```bash
python main.py train --model metrofi --device cpu
```

The MetroFi config sets:

```python
training_mode = "weighted"
```

That makes `main.py` select `DiffusionWeightedGraphModule`. In this mode:

- The model output edge dimension is forced to `E = 1`.
- The loss is masked MSE between predicted edge weights and target adjacency.
- The Jacobi SDE still perturbs edge weights in `[0, 1]`.
- Validation logs weighted MSE and sampled wireless graphs.
- EMA is enabled by default.

Checkpoints are written under:

```text
checkpoints/metrofi/
```

At fit end, the code saves:

```text
checkpoints/metrofi/weights.pth
checkpoints/metrofi/weights_ema.pth
```

Lightning checkpoints are also saved according to
`cfg.general.save_checkpoint_every_n_epochs`.

## Generation And Evaluation

Generate and evaluate with:

```bash
python main.py gen --model metrofi --device cuda:0
```

Or use a specific checkpoint:

```bash
python main.py gen \
  --model metrofi \
  --device cuda:0 \
  --checkpoint checkpoints/metrofi/weights_ema.pth \
  --num-samples 128
```

The MetroFi generation path is `run_gen_wireless` in `main.py`. It:

1. Loads the MetroFi dataset and metadata.
2. Forces `training_mode = "weighted"`.
3. Sets the sampler node count to the fixed AP universe from metadata.
4. Loads `weights_ema.pth` if present, otherwise `weights.pth`.
5. Samples full weighted adjacency matrices.
6. Rescales generated weights from `[0, 1]` back to the dataset interference range.
7. Converts dense weighted adjacencies to NetworkX graphs.
8. Computes wireless metrics against the test split.
9. Saves plots under `samples/`.

Expected generated artifacts include:

```text
samples/wireless.png
samples/wireless_edge_weight_hist_full.png
samples/wireless_edge_weight_hist_subgraphs.png
samples/wireless_weight_heatmaps_full.png
```

The sampler currently also writes reverse-process diagnostic figures to:

```text
tests/history_graphs.png
tests/history_heatmaps.png
```

That happens inside `PCSolver.solve`, so it is part of generation, not a unit
test.

## Wireless Metrics

`WirelessSamplingMetrics` computes:

- `edge_ks`: mean KS statistic over edge-wise interference distributions
- `edge_wasserstein`: mean Wasserstein distance over edge-wise interference distributions
- `edge_pairs_used`: number of AP pairs with both reference and generated values
- `degree_weighted`: MMD over weighted degree histograms
- `spectre`: spectral MMD on sampled subgraphs

For structural metrics, generated full graphs are sampled down to the empirical
test graph size distribution. This tries to compare generated full AP-universe
graphs to the smaller observed-location subgraphs in the reference set.

## Important Caveats

- The MetroFi raw and processed data are not included in this checkout.
- `src/dataset/wireless.py` error text says to run `tools/build_metrofi_dataset.py`,
  but the actual script in this repo is
  `src/dataset_generation/gen_metrofi_dataset.py`.
- Training uses normalized weights, while some evaluation plots and metrics use
  `interference_raw`. Keep this distinction clear when changing evaluation.
- Generated wireless graphs are dense if `keep_zero_weights=True`; metrics that
  iterate over all edges may include zero-weight generated edges.
- The config assumes a 70-node AP universe, but generation can override this
  from dataset metadata via `datamodule.num_mac_addresses()`.

## Likely Files To Modify Next

For training changes:

- `configs/config_metrofi.py`
- `src/train/trainer_graph.py`, especially `DiffusionWeightedGraphModule`
- `src/train/base_module.py`, for validation sampling and logging

For evaluation changes:

- `main.py`, function `run_gen_wireless`
- `src/metrics/val.py`, class `WirelessSamplingMetrics`
- `src/visualization/plots.py`, for wireless plots

For dataset changes:

- `src/dataset_generation/gen_metrofi_dataset.py`
- `src/dataset/wireless.py`

## Training Job Commands

Debug conditional MetroFi run. This uses `configs/config_metrofi_debug.py`: 64 train graphs, 16 validation graphs, 16 test graphs, batch size 8, 20 diffusion steps, and `conditional=True`. It writes to `checkpoints/metrofi-debug-cond`.

```bash
python csub.py -n debug-metrofi-cond --train --node_type h200 -g 0.3 --command "
conda activate graphon;
cd /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion;
python main.py train --model metrofi_debug --device cuda:0
"
```

Official unconditional MetroFi training. This uses `configs/config_metrofi.py`: weighted training only, `conditional=False`, `positional_encoding=False`. It writes to `checkpoints/metrofi-uncond`.

```bash
python csub.py -n train-metrofi-uncond-bs128 --train --node_type h200 -g 0.3 --command "
conda activate graphon;
cd /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion;
python main.py train --model metrofi_uncond --device cuda:0
"
```

Official conditional MetroFi training. This uses `configs/config_metrofi_cond.py`: location condition, node positional encoding, CFG, and conditional masked MSE/MAE validation. It writes to `checkpoints/metrofi-cond`.

```bash
python csub.py -n train-metrofi-cond-bs128 --train --node_type h200 -g 0.3 --command "
conda activate graphon;
cd /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion;
python main.py train --model metrofi_cond --device cuda:0
"
```

Official conditional generation/evaluation after training:

```bash
python csub.py -n eval-metrofi-cond --train --node_type h200 -g 0.3 --command "
conda activate graphon;
cd /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion;
python main.py gen --model metrofi_cond --device cuda:0
"
```
