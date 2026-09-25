# MIDAR: Surrogate LiDAR Detection Model for Microscopic Traffic Simulators

MIDAR is a graph-transformer-based surrogate sensor model that mimics realistic LiDAR object detection using only vehicle-level features available in microscopic traffic simulators (e.g., SUMO). It bridges the gap between the scalability of microscopic traffic simulators and the perception fidelity of game-engine-based simulators (e.g., CARLA).

## Repository Structure

```
MIDAR/
├── los_graphormer/              # Core model package
│   ├── models/
│   │   ├── los_graphormer.py    # LoS-Graphormer model
│   │   └── baselines.py         # MLP, GCN, Vanilla Transformer baselines
│   ├── data/
│   │   ├── carla_dataset.py     # CARLA dataset loader
│   │   └── nuscenes_dataset.py  # nuScenes dataset loader
│   └── utils/
│       └── training.py          # Training loop, evaluation, losses
├── train_carla.py               # Training script (CARLA)
├── train_nuscenes.py            # Training script (nuScenes)
├── data/
│   ├── dataset_MIDAR_carla.csv  # CARLA dataset
│   └── dataset_MIDAR_nuscenes.csv  # nuScenes dataset
├── trained_model/               # <detector>/<dataset>/<model>.pth
│   ├── centerpoint/             # Trained on CenterPoint labels (paper models)
│   │   ├── carla/
│   │   └── nuscenes/
│   └── bevfusion/               # Trained on BEVFusion labels
│       └── nuscenes/
├── application_evaluation/
│   ├── adaptive_tsc/            # CP-based adaptive signal control
│   ├── trajectory_reconstruction/  # Vehicle trajectory reconstruction
│   └── computational_cost/      # Runtime benchmarking with SUMO
└── run.sh                       # Batch training script
```

## Requirements

**Core:**
- Python 3.8+
- PyTorch (with CUDA support recommended)
- PyTorch Geometric
- NumPy, Pandas, scikit-learn

**For application evaluations:**
- Shapely, SciPy, Matplotlib, statsmodels
- Pyomo (trajectory reconstruction optimization)
- psutil, pynvml (computational cost benchmarking)
- SUMO with TraCI (traffic simulation applications)

### Environment Setup

Create a conda environment with all dependencies:

```bash
conda create -n midar python=3.10 -y
conda activate midar

# Install PyTorch (adjust CUDA version as needed, see https://pytorch.org)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install PyTorch Geometric
conda install pyg -c pyg -y

# Install remaining dependencies
pip install numpy pandas scikit-learn scipy matplotlib shapely statsmodels pyomo psutil pynvml
```

> **Note:** For SUMO/TraCI integration (adaptive signal control and computational cost benchmarks), install [SUMO](https://sumo.dlr.de/docs/Installing/index.html) separately and ensure `traci` is available on your Python path.

## Quick Start

### Training

Train the LoS-Graphormer (MIDAR) on CARLA data:

```bash
python train_carla.py \
  --csv-path data/dataset_MIDAR_carla.csv \
  --use-ray-hit \
  --model-type los_graphormer \
  --ckpt-path runs/centerpoint/carla/los_graphormer_5F.pth
```

Train on nuScenes data:

```bash
python train_nuscenes.py \
  --csv-path data/dataset_MIDAR_nuscenes.csv \
  --use-ray-hit \
  --model-type los_graphormer \
  --ckpt-path runs/centerpoint/nuscenes/los_graphormer_5F.pth
```

> **Note:** Point `--ckpt-path` outside `trained_model/` (e.g. `runs/`) when training. The pre-trained checkpoints use the same file names, so training into `trained_model/` overwrites them. The output directory is created automatically.

### Input Features

| Variant | Flag            | Per-vehicle features         |
|---------|-----------------|------------------------------|
| **5F**  | `--use-ray-hit` | `[dist, ray_hit, w, l, h]`   |
| **4F**  | *(omit)*        | `[dist, w, l, h]`            |

5F is the full MIDAR model; 4F is the ablation without the ray-hit feature.

### Model Options

| `--model-type`     | Description                          |
|--------------------|--------------------------------------|
| `los_graphormer`   | LoS-Graphormer                       |
| `vanilla`          | Vanilla Transformer baseline         |
| `gcn`              | GCN baseline                         |
| `mlp`              | MLP baseline                         |

### Key Arguments

| Argument          | Default | Description                              |
|-------------------|---------|------------------------------------------|
| `--use-ray-hit`   | off     | Include ray-hit feature                  |
| `--d-model`       | 128     | Transformer hidden dimension             |
| `--nhead`         | 4       | Number of attention heads                |
| `--num-layers`    | 3       | Number of transformer layers             |
| `--batch-size`    | 1       | Batch size                               |
| `--lr`            | 2e-4    | Learning rate                            |
| `--patience`      | 10      | Early stopping patience                  |
| `--max-epochs`    | 200     | Maximum training epochs                  |
| `--seed`          | 18      | Random seed (also sets the train/val/test scene split) |

## Pre-trained Models

Checkpoints are organized as `trained_model/<detector>/<dataset>/<model>.pth`, where `<detector>` is the LiDAR detector whose outputs MIDAR was trained to mimic:

```
trained_model/
├── centerpoint/                 # Paper models
│   ├── carla/
│   └── nuscenes/
└── bevfusion/
    └── nuscenes/
```

Every `<detector>/<dataset>/` folder contains the same five checkpoints:

| File                      | Model                            | `--model-type`   | Train with        |
|---------------------------|----------------------------------|------------------|-------------------|
| `los_graphormer_5F.pth`   | LoS-Graphormer (MIDAR), ray-hit  | `los_graphormer` | `--use-ray-hit`   |
| `los_graphormer_4F.pth`   | LoS-Graphormer, no ray-hit       | `los_graphormer` |                   |
| `vanilla_transformer.pth` | Vanilla Transformer baseline     | `vanilla`        | `--use-ray-hit`   |
| `mlp.pth`                 | MLP baseline                     | `mlp`            | `--use-ray-hit`   |
| `gcn.pth`                 | GCN baseline                     | `gcn`            | `--use-ray-hit`   |

### Test-set AUC (CenterPoint)

| Model                  | CARLA  | nuScenes |
|------------------------|--------|----------|
| LoS-Graphormer (5F)    | 0.9385 | 0.8647   |
| LoS-Graphormer (4F)    | 0.8982 | 0.8665   |
| Vanilla Transformer    | 0.9135 | 0.8378   |
| MLP                    | 0.8629 | 0.8285   |
| GCN                    | 0.8352 | 0.7967   |

## Application Evaluations

Two ITS applications demonstrate MIDAR's practical value:

- **Adaptive Traffic Signal Control** (`application_evaluation/adaptive_tsc/`) — CP-based signal control integrated with SUMO, comparing MIDAR against perfect detection and random-drop baselines.

- **Vehicle Trajectory Reconstruction** (`application_evaluation/trajectory_reconstruction/`) — Reconstructing complete vehicle trajectories from partial cooperative perception data generated by MIDAR.

- **Computational Cost** (`application_evaluation/computational_cost/`) — Runtime benchmarking showing MIDAR introduces minimal overhead for real-time SUMO integration.

<!-- ## Citation

If you use MIDAR in your research, please cite:

```bibtex
@article{zhu2025midar,
  title={Empowering Microscopic Traffic Simulators with Realistic Perception using Surrogate Sensor Models},
  author={Zhu, Tianheng and Feng, Yiheng},
  year={2025}
}
``` -->

## License

This project is licensed under the [MIT License](LICENSE).
