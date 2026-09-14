import os
import torch

# Hardware configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset parameter limits
MAX_EPISODES = 1693  
JSON_GT_PATH = "/kaggle/input/datasets/saydobidkhusanov/atomicvlagt/libero_lerobot.json"

# Visual foundation backbone model identifiers
STATIC_MODEL_ID = "google/siglip-base-patch16-224"
WRIST_MODEL_ID = "facebook/dino-vitb16"

# Green AI benchmark evaluation coefficients
GPU_TDP_WATTS = 350.0
PUE_FACTOR = 1.58
CARBON_INTENSITY = 0.385

# Projector dimension constants
STATIC_DIM = 768
WRIST_DIM = 768