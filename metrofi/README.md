# MetroFi Clean Conditional Commands

This file keeps only the commands needed for the clean MetroFi conditional run. The clean setting is different from the first MetroFi conditional checkpoint we trained: the old checkpoint used `use_sampled_features=True`, while the current config uses `use_sampled_features=False`.

Why this matters: MetroFi edges are continuous weights in `[0, 1]`, not binary edge indicators. With sampled features enabled, the feature extractor receives a Bernoulli-sampled version of the noisy adjacency, which discards edge magnitude information. For MetroFi, we want both training and sampling to use the continuous weighted adjacency directly.

Current config assumptions:

- model: `metrofi_cond`
- output folder/checkpoints: `metrofi-cond-nosf`
- conditional generation: enabled with location coordinates and node positional encoding
- `use_sampled_features=False` in config
- generation commands also explicitly pass `--no-model-use-sampled-features`
- validation sweep uses 128 validation graphs for speed
- final test uses 943 test graphs

The validation sweep commands below use `/mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion/checkpoints/metrofi-cond-nosf/epoch499.ckpt`. The recommended test command uses `/mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion/checkpoints/metrofi-cond-nosf/epoch999.ckpt` with the best validation setting.

## 1. Training

This starts the clean conditional MetroFi training from `configs/config_metrofi_cond.py`. It writes checkpoints and final weights under `checkpoints/metrofi-cond-nosf/`, and results under `results/metrofi-cond-nosf/`. The GPU request matches the previous MetroFi conditional training run.

python csub.py -n train-metrofi-cond-nosf-bs256 --train --node_type h200 -g 0.2 --command "
conda activate graphon;
cd /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion;
python main.py train --model metrofi_cond --device cuda:0 --batch-size 256
"

## 2. Hyperparameter Search On 128 Validation Graphs

Run this after training finishes and after you fill in the checkpoint path. The sweep is intentionally local, not broad: the previous best region was `guidance=2.0`, `predictor=milstein`, `eps_time=0.05`, default corrector. Since CFG guidance often has the largest effect, most commands vary `guidance_scale` around that point. The remaining commands probe `eps_time`, `heun`, and `em` as sanity checks.

All search commands use:

- `--conditional-eval-split val`
- `--conditional-eval-graphs 128`
- `--conditional-eval-condition-mode true`
- `--no-model-use-sampled-features`
- `--batch-size 256`
- `--sampler-snr 0.01 --sampler-scale-eps 0.1 --sampler-n-steps 1`
- `-g 0.1`

How to choose the best validation run:

- Primary if we care about histogram matching: lower `edge_weight_pooled_wasserstein` and `edge_weight_pooled_hist_js`.
- Secondary reconstruction checks: lower `mse` and `mae`.
- Also open the saved `pooled_edge_weight_hist.png` for the promising runs.

Validation outputs are saved under `results/metrofi-cond-nosf/conditional_val/`, and the summary line is appended to `results/metrofi-cond-nosf/conditional_val/metrics.txt`.

python csub.py -n mf-nosf-local-01-mil-g12-e03 --train --node_type h200 -g 0.1 --command "
conda activate graphon;
cd /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion;
python main.py gen --model metrofi_cond --device cuda:0 --checkpoint /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion/checkpoints/metrofi-cond-nosf/epoch499.ckpt --conditional-eval-only --conditional-eval-split val --conditional-eval-graphs 128 --conditional-eval-condition-mode true --no-model-use-sampled-features --batch-size 256 --sampler-snr 0.01 --sampler-scale-eps 0.1 --sampler-n-steps 1 --conditional-eval-name local-nosf-milstein-g12-e03 --guidance-scale 1.2 --sampler-predictor milstein --sampler-eps-time 0.03
"

python csub.py -n mf-nosf-local-02-mil-g12-e05 --train --node_type h200 -g 0.1 --command "
conda activate graphon;
cd /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion;
python main.py gen --model metrofi_cond --device cuda:0 --checkpoint /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion/checkpoints/metrofi-cond-nosf/epoch499.ckpt --conditional-eval-only --conditional-eval-split val --conditional-eval-graphs 128 --conditional-eval-condition-mode true --no-model-use-sampled-features --batch-size 256 --sampler-snr 0.01 --sampler-scale-eps 0.1 --sampler-n-steps 1 --conditional-eval-name local-nosf-milstein-g12-e05 --guidance-scale 1.2 --sampler-predictor milstein --sampler-eps-time 0.05
"

python csub.py -n mf-nosf-local-03-mil-g15-e03 --train --node_type h200 -g 0.1 --command "
conda activate graphon;
cd /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion;
python main.py gen --model metrofi_cond --device cuda:0 --checkpoint /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion/checkpoints/metrofi-cond-nosf/epoch499.ckpt --conditional-eval-only --conditional-eval-split val --conditional-eval-graphs 128 --conditional-eval-condition-mode true --no-model-use-sampled-features --batch-size 256 --sampler-snr 0.01 --sampler-scale-eps 0.1 --sampler-n-steps 1 --conditional-eval-name local-nosf-milstein-g15-e03 --guidance-scale 1.5 --sampler-predictor milstein --sampler-eps-time 0.03
"

python csub.py -n mf-nosf-local-04-mil-g15-e05 --train --node_type h200 -g 0.1 --command "
conda activate graphon;
cd /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion;
python main.py gen --model metrofi_cond --device cuda:0 --checkpoint /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion/checkpoints/metrofi-cond-nosf/epoch499.ckpt --conditional-eval-only --conditional-eval-split val --conditional-eval-graphs 128 --conditional-eval-condition-mode true --no-model-use-sampled-features --batch-size 256 --sampler-snr 0.01 --sampler-scale-eps 0.1 --sampler-n-steps 1 --conditional-eval-name local-nosf-milstein-g15-e05 --guidance-scale 1.5 --sampler-predictor milstein --sampler-eps-time 0.05
"

python csub.py -n mf-nosf-local-05-mil-g18-e03 --train --node_type h200 -g 0.1 --command "
conda activate graphon;
cd /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion;
python main.py gen --model metrofi_cond --device cuda:0 --checkpoint /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion/checkpoints/metrofi-cond-nosf/epoch499.ckpt --conditional-eval-only --conditional-eval-split val --conditional-eval-graphs 128 --conditional-eval-condition-mode true --no-model-use-sampled-features --batch-size 256 --sampler-snr 0.01 --sampler-scale-eps 0.1 --sampler-n-steps 1 --conditional-eval-name local-nosf-milstein-g18-e03 --guidance-scale 1.8 --sampler-predictor milstein --sampler-eps-time 0.03
"

python csub.py -n mf-nosf-local-06-mil-g18-e05 --train --node_type h200 -g 0.1 --command "
conda activate graphon;
cd /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion;
python main.py gen --model metrofi_cond --device cuda:0 --checkpoint /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion/checkpoints/metrofi-cond-nosf/epoch499.ckpt --conditional-eval-only --conditional-eval-split val --conditional-eval-graphs 128 --conditional-eval-condition-mode true --no-model-use-sampled-features --batch-size 256 --sampler-snr 0.01 --sampler-scale-eps 0.1 --sampler-n-steps 1 --conditional-eval-name local-nosf-milstein-g18-e05 --guidance-scale 1.8 --sampler-predictor milstein --sampler-eps-time 0.05
"

python csub.py -n mf-nosf-local-07-mil-g20-e03 --train --node_type h200 -g 0.1 --command "
conda activate graphon;
cd /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion;
python main.py gen --model metrofi_cond --device cuda:0 --checkpoint /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion/checkpoints/metrofi-cond-nosf/epoch499.ckpt --conditional-eval-only --conditional-eval-split val --conditional-eval-graphs 128 --conditional-eval-condition-mode true --no-model-use-sampled-features --batch-size 256 --sampler-snr 0.01 --sampler-scale-eps 0.1 --sampler-n-steps 1 --conditional-eval-name local-nosf-milstein-g20-e03 --guidance-scale 2.0 --sampler-predictor milstein --sampler-eps-time 0.03
"

python csub.py -n mf-nosf-local-08-mil-g20-e05 --train --node_type h200 -g 0.1 --command "
conda activate graphon;
cd /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion;
python main.py gen --model metrofi_cond --device cuda:0 --checkpoint /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion/checkpoints/metrofi-cond-nosf/epoch499.ckpt --conditional-eval-only --conditional-eval-split val --conditional-eval-graphs 128 --conditional-eval-condition-mode true --no-model-use-sampled-features --batch-size 256 --sampler-snr 0.01 --sampler-scale-eps 0.1 --sampler-n-steps 1 --conditional-eval-name local-nosf-milstein-g20-e05 --guidance-scale 2.0 --sampler-predictor milstein --sampler-eps-time 0.05
"

python csub.py -n mf-nosf-local-09-mil-g22-e03 --train --node_type h200 -g 0.1 --command "
conda activate graphon;
cd /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion;
python main.py gen --model metrofi_cond --device cuda:0 --checkpoint /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion/checkpoints/metrofi-cond-nosf/epoch499.ckpt --conditional-eval-only --conditional-eval-split val --conditional-eval-graphs 128 --conditional-eval-condition-mode true --no-model-use-sampled-features --batch-size 256 --sampler-snr 0.01 --sampler-scale-eps 0.1 --sampler-n-steps 1 --conditional-eval-name local-nosf-milstein-g22-e03 --guidance-scale 2.2 --sampler-predictor milstein --sampler-eps-time 0.03
"

python csub.py -n mf-nosf-local-10-mil-g22-e05 --train --node_type h200 -g 0.1 --command "
conda activate graphon;
cd /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion;
python main.py gen --model metrofi_cond --device cuda:0 --checkpoint /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion/checkpoints/metrofi-cond-nosf/epoch499.ckpt --conditional-eval-only --conditional-eval-split val --conditional-eval-graphs 128 --conditional-eval-condition-mode true --no-model-use-sampled-features --batch-size 256 --sampler-snr 0.01 --sampler-scale-eps 0.1 --sampler-n-steps 1 --conditional-eval-name local-nosf-milstein-g22-e05 --guidance-scale 2.2 --sampler-predictor milstein --sampler-eps-time 0.05
"

python csub.py -n mf-nosf-local-11-mil-g25-e03 --train --node_type h200 -g 0.1 --command "
conda activate graphon;
cd /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion;
python main.py gen --model metrofi_cond --device cuda:0 --checkpoint /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion/checkpoints/metrofi-cond-nosf/epoch499.ckpt --conditional-eval-only --conditional-eval-split val --conditional-eval-graphs 128 --conditional-eval-condition-mode true --no-model-use-sampled-features --batch-size 256 --sampler-snr 0.01 --sampler-scale-eps 0.1 --sampler-n-steps 1 --conditional-eval-name local-nosf-milstein-g25-e03 --guidance-scale 2.5 --sampler-predictor milstein --sampler-eps-time 0.03
"

python csub.py -n mf-nosf-local-12-mil-g25-e05 --train --node_type h200 -g 0.1 --command "
conda activate graphon;
cd /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion;
python main.py gen --model metrofi_cond --device cuda:0 --checkpoint /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion/checkpoints/metrofi-cond-nosf/epoch499.ckpt --conditional-eval-only --conditional-eval-split val --conditional-eval-graphs 128 --conditional-eval-condition-mode true --no-model-use-sampled-features --batch-size 256 --sampler-snr 0.01 --sampler-scale-eps 0.1 --sampler-n-steps 1 --conditional-eval-name local-nosf-milstein-g25-e05 --guidance-scale 2.5 --sampler-predictor milstein --sampler-eps-time 0.05
"

python csub.py -n mf-nosf-local-13-heun-g08-e05 --train --node_type h200 -g 0.1 --command "
conda activate graphon;
cd /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion;
python main.py gen --model metrofi_cond --device cuda:0 --checkpoint /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion/checkpoints/metrofi-cond-nosf/epoch499.ckpt --conditional-eval-only --conditional-eval-split val --conditional-eval-graphs 128 --conditional-eval-condition-mode true --no-model-use-sampled-features --batch-size 256 --sampler-snr 0.01 --sampler-scale-eps 0.1 --sampler-n-steps 1 --conditional-eval-name local-nosf-heun-g08-e05 --guidance-scale 0.8 --sampler-predictor heun --sampler-eps-time 0.05
"

python csub.py -n mf-nosf-local-14-heun-g10-e05 --train --node_type h200 -g 0.1 --command "
conda activate graphon;
cd /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion;
python main.py gen --model metrofi_cond --device cuda:0 --checkpoint /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion/checkpoints/metrofi-cond-nosf/epoch499.ckpt --conditional-eval-only --conditional-eval-split val --conditional-eval-graphs 128 --conditional-eval-condition-mode true --no-model-use-sampled-features --batch-size 256 --sampler-snr 0.01 --sampler-scale-eps 0.1 --sampler-n-steps 1 --conditional-eval-name local-nosf-heun-g10-e05 --guidance-scale 1.0 --sampler-predictor heun --sampler-eps-time 0.05
"

python csub.py -n mf-nosf-local-15-heun-g12-e05 --train --node_type h200 -g 0.1 --command "
conda activate graphon;
cd /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion;
python main.py gen --model metrofi_cond --device cuda:0 --checkpoint /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion/checkpoints/metrofi-cond-nosf/epoch499.ckpt --conditional-eval-only --conditional-eval-split val --conditional-eval-graphs 128 --conditional-eval-condition-mode true --no-model-use-sampled-features --batch-size 256 --sampler-snr 0.01 --sampler-scale-eps 0.1 --sampler-n-steps 1 --conditional-eval-name local-nosf-heun-g12-e05 --guidance-scale 1.2 --sampler-predictor heun --sampler-eps-time 0.05
"

python csub.py -n mf-nosf-local-16-heun-g15-e05 --train --node_type h200 -g 0.1 --command "
conda activate graphon;
cd /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion;
python main.py gen --model metrofi_cond --device cuda:0 --checkpoint /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion/checkpoints/metrofi-cond-nosf/epoch499.ckpt --conditional-eval-only --conditional-eval-split val --conditional-eval-graphs 128 --conditional-eval-condition-mode true --no-model-use-sampled-features --batch-size 256 --sampler-snr 0.01 --sampler-scale-eps 0.1 --sampler-n-steps 1 --conditional-eval-name local-nosf-heun-g15-e05 --guidance-scale 1.5 --sampler-predictor heun --sampler-eps-time 0.05
"

python csub.py -n mf-nosf-local-17-em-g15-e05 --train --node_type h200 -g 0.1 --command "
conda activate graphon;
cd /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion;
python main.py gen --model metrofi_cond --device cuda:0 --checkpoint /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion/checkpoints/metrofi-cond-nosf/epoch499.ckpt --conditional-eval-only --conditional-eval-split val --conditional-eval-graphs 128 --conditional-eval-condition-mode true --no-model-use-sampled-features --batch-size 256 --sampler-snr 0.01 --sampler-scale-eps 0.1 --sampler-n-steps 1 --conditional-eval-name local-nosf-em-g15-e05 --guidance-scale 1.5 --sampler-predictor em --sampler-eps-time 0.05
"

python csub.py -n mf-nosf-local-18-em-g20-e05 --train --node_type h200 -g 0.1 --command "
conda activate graphon;
cd /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion;
python main.py gen --model metrofi_cond --device cuda:0 --checkpoint /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion/checkpoints/metrofi-cond-nosf/epoch499.ckpt --conditional-eval-only --conditional-eval-split val --conditional-eval-graphs 128 --conditional-eval-condition-mode true --no-model-use-sampled-features --batch-size 256 --sampler-snr 0.01 --sampler-scale-eps 0.1 --sampler-n-steps 1 --conditional-eval-name local-nosf-em-g20-e05 --guidance-scale 2.0 --sampler-predictor em --sampler-eps-time 0.05
"


## 3. Epoch999 Test Command

Run this for the final test on `epoch999.ckpt`. The command uses the current best setting: `guidance=1.8`, `predictor=milstein`, `eps_time=0.03`, with `use_sampled_features=False`.

The test run evaluates 943 test graphs and saves `conditional_samples.pt`, `metrics.json`, `comparison.png`, and `pooled_edge_weight_hist.png` under `results/metrofi-cond-nosf/conditional_test/`. The saved tensor file lets us recompute metrics later without rerunning sampling.

python csub.py -n eval-metrofi-test-nosf-best-e999-g18-e03 --train --node_type h200 -g 0.1 --command "
conda activate graphon;
cd /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion;
python main.py gen --model metrofi_cond --device cuda:0 --checkpoint /mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion/checkpoints/metrofi-cond-nosf/epoch999.ckpt --conditional-eval-only --conditional-eval-split test --conditional-eval-graphs 943 --conditional-eval-name test-nosf-best-e999-g18-milstein-e03 --conditional-eval-condition-mode true --guidance-scale 1.8 --sampler-predictor milstein --sampler-eps-time 0.03 --sampler-snr 0.01 --sampler-scale-eps 0.1 --sampler-n-steps 1 --no-model-use-sampled-features --batch-size 256
"
