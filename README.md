# DiPhon

This repository contains the implementation of Jacobi Graph Diffusion (JGD), a framework for generating graphs via diffusion models.

## Installation

You can install the dependencies using either `conda` or `pip`.

### Using Conda (Recommended)
An `environment.yml` file is provided to create a conda environment named `graphon` with all necessary dependencies:
```bash
conda env create -f environment.yml
conda activate graphon
```

### Using Pip
Alternatively, you can install the required packages using `pip` and the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Running the Code

### Training
To train the diffusion model on a specific dataset (e.g., Stochastic Block Models with 2 communities, Preferential Attachment, Trees), run the corresponding training script:
```bash
# Train on SBM with 2 communities
python train_sbm_2comms.py

# Train on Preferential Attachment graphs
python train_pa.py

# Train on Tree graphs
python train_tree.py
```

Checkpoints will be saved automatically to the `checkpoints/` directory.

### Generation and Evaluation
To generate new graphs from a trained model and optionally evaluate them against the reference datasets, use the corresponding generation scripts. 

You can configure the generation process with various arguments such as `--min-nodes`, `--max-nodes`, and `--num-samples`.
```bash
# Generate SBM graphs
python gen_sbm_2comms.py --num-samples 100 --save-graphs-path "samples_sbm.pkl"

# Generate PA graphs
python gen_pa.py --min-nodes 100 --max-nodes 100 --num-samples 500 --save-graphs-path "samples_pa.pkl"

# Generate Tree graphs
python gen_tree.py
```

If you want to load a specific model checkpoint, you can pass the `--checkpoint` flag:
```bash
python gen_pa.py --checkpoint checkpoints/pa_graphon/epoch=16999.ckpt
```

To run a batch generation for out-of-distribution sizes, you can execute the provided shell script:
```bash
bash run_ood_gen.sh
```