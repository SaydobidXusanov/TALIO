# TALIO
TALIO: An Unsupervised Temporal Segmentation Method for Robotic Sub-Action Discovery

Official repository for **TALIO**, an unsupervised, training-free method for discovering temporal boundaries and sub-actions in demonstration trajectories, enabling lightweight action regularization for robotic policy learning.

---

## Repository Structure

```text
TALIO/
├── Action Regularization/
│   ├── TALIO-LeRobot/
│   │   └── action_reg_TALIO_LeRobot.py
│   └── TALIO-RLDS/
│       └── action_reg_TALIO_RLDS.py
├── TALIO Method/
│   ├── config.py
│   ├── dataset.py
│   ├── main.py
│   ├── models.py
│   ├── segmentation.py
│   └── vla_comprehensive_ablation_matrix.csv
├── Sub-Action Videos/
│   └── cluster_0_short_60s.mp4 ... cluster_9_short_60s.mp4
├── LIBERO_unsupervised_sub_skills.json
├── LICENSE
└── README.md

```

---

## Overview

TALIO segments robotic manipulation trajectories into discrete sub-action intervals without human annotations or heavy Vision-Language Models. The discovered sub-actions can be directly integrated into dataset pipelines (e.g., LeRobot, RLDS) to regularize continuous action sequences during model training.

Key highlights:

* **Training-free & Lightweight:** Low compute overhead (computes boundaries across thousands of trajectories using minimal VRAM).
* **Unsupervised Sub-Action Discovery:** Automatically extracts temporal boundary indices (`start`, `end`), cluster assignments (`cluster_id`), and segment-mean action vectors (`mean_action`).
* **Easy Integration:** Ready-to-use dataset wrappers for PyTorch (LeRobot) and TensorFlow (RLDS) pipelines.

---

## Prerequisites

The main TALIO segmentation pipeline (`TALIO Method/main.py`) streams and processes the standard LIBERO dataset directly from Hugging Face within the execution code without requiring prior manual downloads.

* Dataset Source: [HuggingFaceVLA/libero](https://huggingface.co/datasets/HuggingFaceVLA/libero)
* Baseline Environment & Dataset Reference: [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)

---

## Pre-Computed Registry Format

`LIBERO_unsupervised_sub_skills.json` maps episode indices to lists of segmented sub-actions:

```json
{
  "0": [
    {
      "start": 0,
      "end": 2,
      "cluster_id": 2,
      "mean_action": [-0.0533, 0.0070, 0.6783, 3.1407, 0.0017, -0.0898, ...]
    },
    {
      "start": 2,
      "end": 13,
      "cluster_id": 5,
      "mean_action": [-0.0533, -0.0004, 0.6767, 3.1398, -0.0014, -0.0907, ...]
    }
  ]
}

```

---

## Usage

### 1. Generating Temporal Boundaries

Run the main pipeline directly to fetch trajectories, extract features, and compute sub-action segmentations:

```bash
python "TALIO Method/main.py"

```

### 2. Action Regularization in Datasets

#### PyTorch / LeRobot

Wrap your existing `LeRobotDataset` using `TALIOCompressedLeRobotDataset` in `Action Regularization/TALIO-LeRobot/action_reg_TALIO_LeRobot.py`:

```python
from Action_Regularization.TALIO_LeRobot.lerobot_dataset_pretrain_mp import TALIOCompressedLeRobotDataset

dataset = TALIOCompressedLeRobotDataset(
    base_dataset=base_lerobot_dataset,
    talio_json_path="LIBERO_unsupervised_sub_skills.json"
)

```

#### TensorFlow / RLDS

Wrap your iterable `RLDSDataset` using `TALIOCompressedDataset` from `Action Regularization/TALIO-RLDS/action_reg_TALIO_RLDS.py`:

```python
from Action_Regularization.TALIO_RLDS.action_reg_TALIO import TALIOCompressedDataset

dataset = TALIOCompressedDataset(
    base_rlds_dataset=base_rlds_dataset,
    talio_json_path="LIBERO_unsupervised_sub_skills.json"
)

```

---

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/SaydobidXusanov/TALIO/blob/main/LICENSE) file for details.

```

```
