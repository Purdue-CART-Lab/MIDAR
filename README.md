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
| `--seed`          | 42 (CARLA), 18 (nuScenes) | Random seed (also sets the train/val/test scene split) |

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

## Using MIDAR in Your Own Application

MIDAR can replace a LiDAR detector in any simulator or dataset that provides vehicle positions, headings and sizes. At each time step, for each sensing vehicle (ego), MIDAR takes the surrounding vehicles within sensing range and returns the subset the LiDAR detector would detect. Vehicles it does not return are false negatives (missed detections).

### Inputs

For every vehicle, in one global coordinate frame:

| Input      | Format                    | Notes                                                        |
|------------|---------------------------|--------------------------------------------------------------|
| `positions`| `{id: (x, y)}` in m       | Ego and all neighbours                                       |
| `headings` | `{id: yaw}` in rad        | Ego and all neighbours. Direction of travel, counter-clockwise from +x (for SUMO: `math.radians(90 - traci.vehicle.getAngle(v))`). Values with \|yaw\| > 3.2 are treated as degrees |
| `dims`     | `{id: (w, l, h)}` in m    | Neighbours                                                   |
| `z` (optional) | `{id: z}` in m        | Neighbours' box-centre height relative to the ego's LiDAR sensor. If omitted, estimated for a flat road as `h/2 - lidar_height` (`lidar_height=1.75` by default) |

The helper translates and rotates the scene into the ego frame and computes the ray-hit feature for you. Pass `z=` (and `lidar_height=`) as keyword arguments to `_build_los_frame_for_cav` when your simulator provides vehicle heights.

### Example

The frame-building and inference helpers live in `application_evaluation/trajectory_reconstruction/MIDAR_helpers.py`. Run from the repository root:

```python
import sys, math, torch
sys.path.append("application_evaluation/trajectory_reconstruction")
from los_graphormer import LoSGraphormer
from MIDAR_helpers import _build_los_frame_for_cav, los_visible_ids_from_graph

# 1. Load a checkpoint (in_feats=5 for *_5F.pth, 4 for *_4F.pth)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LoSGraphormer(in_feats=5, d_model=128, nhead=4, num_layers=3,
                      dim_feedforward=256, dropout=0.2, max_len=32,
                      num_classes=2).to(device)
model.load_state_dict(torch.load(
    "trained_model/centerpoint/carla/los_graphormer_5F.pth", map_location=device))
model.eval()

SENSING_RANGE = 80.0   # m
OCC_THRESH = 0.272     # vehicle is detected if P(missed) < OCC_THRESH

# 2. At each time step, for each ego vehicle:
def midar_detect(ego_id, positions, headings, dims):
    ex, ey = positions[ego_id]
    neigh = [v for v in positions
             if v != ego_id and math.hypot(positions[v][0] - ex, positions[v][1] - ey) <= SENSING_RANGE]
    if not neigh:
        return []
    # The helper expects integer IDs: map the ego to 0 and neighbours to 1..N
    ids = [ego_id] + neigh
    data = _build_los_frame_for_cav(
        0, list(range(1, len(ids))),
        {i: positions[v] for i, v in enumerate(ids)},
        {i: dims[v] for i, v in enumerate(ids) if i > 0},
        {i: headings[v] for i, v in enumerate(ids)},
        use_ray_hit=True,   # False for *_4F.pth
    ).to(device)
    visible = los_visible_ids_from_graph(data, model, occ_thresh=OCC_THRESH)
    return [ids[i] for i in visible if i != 0]
```

For CP-based applications, run `midar_detect` for every CAV and take the union of the results as the cooperative observation.

### Choosing a Checkpoint and Threshold

Match the checkpoint to the detector you want to mimic and the environment closest to yours. Keep the sensing range at the detection range of the training data. For example:

| Checkpoint                                | Sensing range | `OCC_THRESH` |
|-------------------------------------------|---------------|--------------|
| `centerpoint/carla/los_graphormer_5F`     | 80 m          | 0.272        |
| `bevfusion/nuscenes/los_graphormer_5F`    | 54 m          | 0.217        |

`OCC_THRESH` trades detections for misses: lowering it makes MIDAR miss more vehicles. If you use a different environment, re-tune the threshold on labelled data from that environment.

## Application Evaluations

The paper's application evaluations also serve as complete integration examples:

- **Adaptive Traffic Signal Control** (`application_evaluation/adaptive_tsc/`) — CP-based signal control in SUMO, comparing MIDAR against perfect detection and random-drop baselines. `ISIG_singapore_MIDAR.py` queries MIDAR online through TraCI.

- **Vehicle Trajectory Reconstruction** (`application_evaluation/trajectory_reconstruction/`) — Reconstructing complete vehicle trajectories from partial cooperative perception data generated by MIDAR. `CP_data_generation.py` runs MIDAR offline on a trajectory CSV with columns `Vehicle_ID, Global_Time, Global_X, Global_Y, v_Width, v_Length, v_Height, yaw, CAV`.

- **Computational Cost** (`application_evaluation/computational_cost/`) — Runtime benchmarking showing MIDAR introduces minimal overhead for real-time SUMO integration (`online_detection_MIDAR.py`).

## Citation

If you use MIDAR in your research, please cite:

```bibtex
@article{zhu2027empowering,
  title   = {Empowering microscopic traffic simulators with realistic perception using surrogate sensor models},
  journal = {Transportation Research Part C: Emerging Technologies},
  volume  = {194},
  pages   = {106031},
  year    = {2027},
  issn    = {0968-090X},
  doi     = {https://doi.org/10.1016/j.trc.2026.106031},
  author  = {Tianheng Zhu and Yiheng Feng}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
