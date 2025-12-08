# MIDAR: a graph-transformer based surrogate sensor model that mimics lidar detections

This repo contains the code for MIDAR, a graph transformer (Graphormer-style)
designed for LiDAR-like occlusion modeling on traffic scenes, plus several baselines:

- MLP baseline (per-vehicle)
- GCN baseline on RM-LoS chains
- Vanilla Transformer baseline (no geometric bias)

## Installation

```bash
conda create -n los-graphormer python=3.10
conda activate los-graphormer

pip install -r requirements.txt

# Example Usage

python train_nuscenes.py \
  --csv-path data/dataset_MIDAR_nuscenes.csv \
  --use-ray-hit \
  --model-type los_graphormer \
  --ckpt-path trained_model/nuscenes_los_graphormer.pth


python train_nuscenes.py --csv-path data/dataset_MIDAR_nuscenes.csv --use-ray-hit --model-type mlp --ckpt-path trained_model/nuscenes_mlp.pth
python train_nuscenes.py --csv-path data/dataset_MIDAR_nuscenes.csv --use-ray-hit --model-type gcn --ckpt-path trained_model/nuscenes_gcn.pth
python train_nuscenes.py --csv-path data/dataset_MIDAR_nuscenes.csv --use-ray-hit --model-type vanilla --ckpt-path trained_model/nuscenes_vanilla_transformer.pth

