# MIDAR: Surrogate LiDAR Detection Model for Microscopic Traffic Simulators

MIDAR is a graph-transformer-based surrogate model that mimics realistic LiDAR object detection using only vehicle-level features available in microscopic traffic simulators (e.g., SUMO). It bridges the gap between the scalability of microscopic traffic simulators and the perception fidelity of game-engine-based simulators (e.g., CARLA).

**Paper:** *Empowering Microscopic Traffic Simulators with Realistic Perception using Surrogate Sensor Models*
Tianheng Zhu, Yiheng Feng — Purdue University

## Key Idea

Game-engine-based simulators generate realistic sensor data but are computationally expensive for large-scale simulations. Microscopic traffic simulators are efficient but lack perception modeling. MIDAR closes this gap by predicting whether each surrounding vehicle is a **true positive (TP)** or **false negative (FN)** detection, using lightweight geometric features instead of raw point clouds.

### How It Works

1. **Refined Multi-hop Line-of-Sight (RM-LoS) Graph** — For each target vehicle, MIDAR traces an occlusion chain from the target through intermediate blockers to the ego AV, capturing which vehicles may obstruct the line of sight.

2. **Height-aware Azimuthal Ray Casting (HARC)** — A ray-hit feature approximates LiDAR point coverage by counting how many azimuth rays from the ego AV can reach the target, computed across multiple height slices.

3. **LoS-Graphormer** — A geometry-aware Graph Transformer that processes each LoS chain. Pairwise geometric relations produce an attention bias encoding physical priors (e.g., closer vehicles have stronger occlusion effects).

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
├── trained_model/               # Pre-trained checkpoints
├── application_evaluation/
│   ├── adaptive_tsc/            # CP-based adaptive signal control
│   ├── trajectory_reconstruction/  # Vehicle trajectory reconstruction
│   └── computational_cost/      # Runtime benchmarking with SUMO
└── run.sh                       # Batch training script
```

## Requirements

- Python 3.8+
- PyTorch
- PyTorch Geometric
- NumPy, Pandas, scikit-learn

## Quick Start

### Training

Train the LoS-Graphormer (MIDAR) on CARLA data:

```bash
python train_carla.py \
  --csv-path data/dataset_MIDAR_carla.csv \
  --use-ray-hit \
  --model-type los_graphormer \
  --ckpt-path trained_model/carla_los_graphormer.pth
```

Train on nuScenes data:

```bash
python train_nuscenes.py \
  --csv-path data/dataset_MIDAR_nuscenes.csv \
  --use-ray-hit \
  --model-type los_graphormer \
  --ckpt-path trained_model/nuscenes_los_graphormer.pth
```

### Model Options

| `--model-type`     | Description                          |
|--------------------|--------------------------------------|
| `los_graphormer`   | LoS-Graphormer (MIDAR, recommended)  |
| `vanilla`          | Vanilla Transformer baseline         |
| `gcn`              | GCN baseline                         |
| `mlp`              | MLP baseline                         |

### Key Arguments

| Argument          | Default | Description                              |
|-------------------|---------|------------------------------------------|
| `--use-ray-hit`   | off     | Include ray-hit feature (5F vs 4F)       |
| `--d-model`       | 128     | Transformer hidden dimension             |
| `--nhead`         | 4       | Number of attention heads                |
| `--num-layers`    | 3       | Number of transformer layers             |
| `--batch-size`    | 1       | Batch size                               |
| `--lr`            | 2e-4    | Learning rate                            |
| `--patience`      | 10      | Early stopping patience                  |
| `--max-epochs`    | 200     | Maximum training epochs                  |

### Train All Models

```bash
bash run.sh
```

## Pre-trained Models

Pre-trained checkpoints are in `trained_model/`. The filename suffix encodes the AUC score (x10000):

| Model                          | Dataset  | AUC    |
|--------------------------------|----------|--------|
| LoS-Graphormer (5F, ray-hit)  | CARLA    | 0.9385 |
| LoS-Graphormer (4F, no ray-hit)| CARLA   | 0.8982 |
| Vanilla Transformer           | CARLA    | 0.9135 |
| MLP                           | CARLA    | 0.8629 |
| GCN                           | CARLA    | 0.8352 |
| LoS-Graphormer (5F, ray-hit)  | nuScenes | 0.8647 |
| LoS-Graphormer (4F, no ray-hit)| nuScenes| 0.8665 |
| Vanilla Transformer           | nuScenes | 0.8378 |
| MLP                           | nuScenes | 0.8285 |
| GCN                           | nuScenes | 0.7967 |

## Application Evaluations

Two ITS applications demonstrate MIDAR's practical value:

- **Adaptive Traffic Signal Control** (`application_evaluation/adaptive_tsc/`) — CP-based signal control integrated with SUMO, comparing MIDAR against perfect detection and random-drop baselines.

- **Vehicle Trajectory Reconstruction** (`application_evaluation/trajectory_reconstruction/`) — Reconstructing complete vehicle trajectories from partial cooperative perception data generated by MIDAR.

- **Computational Cost** (`application_evaluation/computational_cost/`) — Runtime benchmarking showing MIDAR introduces minimal overhead for real-time SUMO integration.

## Data Format

The input CSV files contain one row per target vehicle per frame with the following columns:

| Column     | Description                                      |
|------------|--------------------------------------------------|
| `frame_id` | Unique frame identifier                          |
| `label`    | Vehicle class (e.g., Car)                        |
| `x, y, z`  | Relative position to ego AV                     |
| `l, w, h`  | Vehicle dimensions (length, width, height)       |
| `heading`  | Vehicle heading (yaw)                            |
| `volume`   | Traffic volume of the scene                      |
| `veh_id`   | Ego vehicle ID                                   |
| `frame_idx`| Frame index within the scene                     |
| `occluded` | Ground truth: False (TP) or True (FN)            |
| `ray_hit`  | HARC ray-hit feature value                       |

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
