import torch
from tqdm import tqdm
from transformers import SiglipVisionModel, SiglipProcessor, ViTModel, ViTImageProcessor
from datasets import load_dataset
from config import DEVICE, MAX_EPISODES, STATIC_MODEL_ID, WRIST_MODEL_ID
from models import apply_pooling_variant, compute_backbone_flops_per_frame

def stream_and_pool_episodes(green_ctx):
    """
    Streams raw vision-action video tokens natively, bypassing local storage caching 
    and preserving GPU memory structures through direct host CPU arrays.
    """
    print("[STREAMING] Loading network foundation backbones...")
    static_processor = SiglipProcessor.from_pretrained(STATIC_MODEL_ID)
    static_model = SiglipVisionModel.from_pretrained(STATIC_MODEL_ID).to(device=DEVICE, dtype=torch.float16).eval()
    wrist_processor = ViTImageProcessor.from_pretrained(WRIST_MODEL_ID)
    wrist_model = ViTModel.from_pretrained(WRIST_MODEL_ID).to(device=DEVICE, dtype=torch.float16).eval()

    raw_accumulators = {}
    ds = load_dataset("HuggingFaceVLA/libero", split="train", streaming=True)
    ds = ds.select_columns(["observation.images.image", "observation.images.image2", "action", "episode_index"])

    flops_per_frame = compute_backbone_flops_per_frame()
    total_frames_processed = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEVICE)

    with tqdm(total=273465, desc="Streaming & Processing Frames", unit=" frames") as pbar:
        for entry in iter(ds):
            ep_idx = entry["episode_index"]
            if ep_idx >= MAX_EPISODES: 
                break

            img_static = entry["observation.images.image"].convert("RGB")
            img_wrist = entry["observation.images.image2"].convert("RGB")

            with torch.no_grad():
                out_static = static_model(**static_processor(images=img_static, return_tensors="pt").to(device=DEVICE, dtype=torch.float16)).last_hidden_state
                out_wrist = wrist_model(**wrist_processor(images=img_wrist, return_tensors="pt").to(device=DEVICE, dtype=torch.float16)).last_hidden_state

            action_vector = torch.tensor(entry["action"]).float()
            
            if ep_idx not in raw_accumulators:
                raw_accumulators[ep_idx] = {"conv": [], "mean": [], "max": [], "actions": []}

            # Map vectors immediately into storage pipelines
            for mode in ["conv", "mean", "max"]:
                f_static = apply_pooling_variant(out_static, strategy=mode, has_cls_token=False, is_wrist=False)
                f_wrist = apply_pooling_variant(out_wrist, strategy=mode, has_cls_token=True, is_wrist=True)
                v_comb = torch.cat([f_static, f_wrist], dim=-1).cpu()  # Consolidate pooled data onto host memory
                raw_accumulators[ep_idx][mode].append(v_comb)

            raw_accumulators[ep_idx]["actions"].append(action_vector)
            total_frames_processed += 1
            pbar.update(1)

    green_ctx["total_flops"] += (total_frames_processed * flops_per_frame)
    if torch.cuda.is_available():
        green_ctx["peak_memory_bytes"] = max(green_ctx["peak_memory_bytes"], torch.cuda.max_memory_allocated(DEVICE))

    del static_model, wrist_model
    torch.cuda.empty_cache()

    processed_episodes = []
    for k in sorted(raw_accumulators.keys()):
        processed_episodes.append({
            "episode_id": k,
            "conv": torch.stack(raw_accumulators[k]["conv"]),
            "mean": torch.stack(raw_accumulators[k]["mean"]),
            "max": torch.stack(raw_accumulators[k]["max"]),
            "actions": torch.stack(raw_accumulators[k]["actions"])
        })
    return processed_episodes