import os
import json
import numpy as np
import torch
from scipy.signal import find_peaks
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from config import DEVICE, JSON_GT_PATH, MAX_EPISODES

def extract_peaks_from_episode(v_data, candidate_prom):
    """
    Identifies change points along the feature manifold via continuous frame similarity profiles.
    """
    T = v_data.shape[0]
    norm_v = v_data / v_data.norm(dim=-1, keepdim=True)
    similarity_matrix = torch.mm(norm_v, norm_v.t()).cpu().numpy()

    step_similarities = np.array([similarity_matrix[t, t+1] for t in range(T - 1)])
    distance_profile = 1.0 - step_similarities
    peaks, _ = find_peaks(distance_profile, distance=5, prominence=candidate_prom)
    
    proposed_cuts = sorted([int(p + 1) for p in peaks])
    return [0] + proposed_cuts + [T], proposed_cuts

def run_agglomerative_clustering(kinematic_vectors, k_num):
    """
    Clusters sub-action signatures using Ward linkage.
    """
    X = np.array(kinematic_vectors)
    sil_score = -1.0
    labels = None
    if len(X) > k_num and k_num < len(X):
        hc = AgglomerativeClustering(n_clusters=k_num, metric='euclidean', linkage='ward')
        labels = hc.fit_predict(X)
        sil_score = silhouette_score(X, labels, metric='euclidean')
    return labels, sil_score

def load_ground_truth():
    """
    Extracts and maps external AtomicVLA segment boundaries.
    """
    if not os.path.exists(JSON_GT_PATH):
        raise FileNotFoundError(f"Could not locate ground truth file at '{JSON_GT_PATH}'. Ensure it is in the active workspace.")

    print(f"Parsing external ground truth dataset from: {JSON_GT_PATH}")
    with open(JSON_GT_PATH, "r") as f:
        raw_gt_data = json.load(f)

    atomic_vla_gt = {}
    for ep_str, data in raw_gt_data.items():
        ep_id = int(ep_str)
        atomic_vla_gt[ep_id] = [
            (int(seg["start_frame"]), int(seg["end_frame"]), seg["primary_action_verb"])
            for seg in data["segments"]
        ]
    return atomic_vla_gt