import time
import json
import numpy as np
import pandas as pd
import torch
from config import DEVICE, MAX_EPISODES, GPU_TDP_WATTS, PUE_FACTOR, CARBON_INTENSITY
from dataset import stream_and_pool_episodes
from segmentation import load_ground_truth, extract_peaks_from_episode, run_agglomerative_clustering

def run_hierarchical_segment_grouping():
    atomic_vla_gt = load_ground_truth()
    green_analytics = {"start_time": time.time(), "total_flops": 0.0, "peak_memory_bytes": 0}
    
    # Process episodes on-the-fly
    in_memory_episodes = stream_and_pool_episodes(green_analytics)

    # 3D Sweep configuration matrices
    prominence_candidates = [0.0005, 0.0010, 0.0015, 0.0020, 0.0030, 0.0050]
    pooling_strategies = ["conv", "mean", "max"]
    cluster_size_candidates = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    sweep_results = []
    
    total_gt_boundaries = sum([len(v) - 1 for k, v in atomic_vla_gt.items() if int(k) < MAX_EPISODES])
    print(f"\n[ABLATION] Launching grid sweep parameter loops...")

    # --- PART A: PEAK DISCOVERY CROSS-SWEEP ---
    for pool_mode in pooling_strategies:
        for candidate_prom in prominence_candidates:
            total_proposed_segments = 0
            perfectly_pure_segments = 0
            matched_gt_boundaries = 0
            kinematic_vectors = []

            for ep_idx, stream in enumerate(in_memory_episodes):
                v_data = stream[pool_mode].to(DEVICE)
                a_data = stream["actions"]
                
                boundaries, proposed_cuts = extract_peaks_from_episode(v_data, candidate_prom)
                gt_intervals = atomic_vla_gt.get(ep_idx, [])
                gt_transition_frames = [step[1] for step in gt_intervals[:-1]]

                for gt_val in gt_transition_frames:
                    if any(abs(proposed_cut - gt_val) <= 4 for proposed_cut in proposed_cuts):
                        matched_gt_boundaries += 1

                for s_idx in range(len(boundaries) - 1):
                    start, end = boundaries[s_idx], boundaries[s_idx+1]
                    total_proposed_segments += 1

                    is_pure = False
                    for gt_start, gt_end, _ in gt_intervals:
                        if start >= gt_start and end <= gt_end + 2:
                            is_pure = True
                            break
                    if is_pure:
                        perfectly_pure_segments += 1

                    kinematic_signature = a_data[start:end].mean(dim=0)
                    norm_kinematic = kinematic_signature / (kinematic_signature.norm() + 1e-8)
                    kinematic_vectors.append(norm_kinematic.numpy())

            purity_rate = (perfectly_pure_segments / total_proposed_segments) * 100 if total_proposed_segments > 0 else 0.0
            boundary_recall = (matched_gt_boundaries / total_gt_boundaries) * 100 if total_gt_boundaries > 0 else 100.0
            
            for k_num in cluster_size_candidates:
                _, sil_score = run_agglomerative_clustering(kinematic_vectors, k_num)
                sweep_results.append({
                    "Method Archetype": f"Peak Finding ({pool_mode.upper()})",
                    "Prominence / Split Param": candidate_prom,
                    "Target Clusters (K)": k_num,
                    "Total Segments": total_proposed_segments,
                    "Semantic Purity (%)": round(purity_rate, 2),
                    "Boundary Recall (%)": round(boundary_recall, 2),
                    "Action Silhouette Score": round(sil_score, 4)
                })

    # --- PART B: UNIFORM BASELINE SWEEP ---
    for split_count in [3, 5, 8, 12]:
        total_proposed_segments = 0
        perfectly_pure_segments = 0
        matched_gt_boundaries = 0
        kinematic_vectors = []

        for ep_idx, stream in enumerate(in_memory_episodes):
            T = stream["conv"].shape[0]
            a_data = stream["actions"]
            boundaries = np.linspace(0, T, split_count + 1, dtype=int).tolist()
            proposed_cuts = boundaries[1:-1]

            gt_intervals = atomic_vla_gt.get(ep_idx, [])
            gt_transition_frames = [step[1] for step in gt_intervals[:-1]]

            for gt_val in gt_transition_frames:
                if any(abs(proposed_cut - gt_val) <= 4 for proposed_cut in proposed_cuts):
                    matched_gt_boundaries += 1

            for s_idx in range(len(boundaries) - 1):
                start, end = boundaries[s_idx], boundaries[s_idx+1]
                total_proposed_segments += 1

                is_pure = False
                for gt_start, gt_end, _ in gt_intervals:
                    if start >= gt_start and end <= gt_end + 2:
                        is_pure = True
                        break
                if is_pure:
                    perfectly_pure_segments += 1

                kinematic_signature = a_data[start:end].mean(dim=0)
                norm_kinematic = kinematic_signature / (kinematic_signature.norm() + 1e-8)
                kinematic_vectors.append(norm_kinematic.numpy())

        purity_rate = (perfectly_pure_segments / total_proposed_segments) * 100 if total_proposed_segments > 0 else 0.0
        boundary_recall = (matched_gt_boundaries / total_gt_boundaries) * 100 if total_gt_boundaries > 0 else 100.0

        for k_num in cluster_size_candidates:
            _, sil_score = run_agglomerative_clustering(kinematic_vectors, k_num)
            sweep_results.append({
                "Method Archetype": "Uniform Splitting Baseline",
                "Prominence / Split Param": split_count,
                "Target Clusters (K)": k_num,
                "Total Segments": total_proposed_segments,
                "Semantic Purity (%)": round(purity_rate, 2),
                "Boundary Recall (%)": round(boundary_recall, 2),
                "Action Silhouette Score": round(sil_score, 4)
            })

    # Save complete structural matrix to disk
    df_sweep = pd.DataFrame(sweep_results)
    df_sweep.to_csv("vla_comprehensive_ablation_matrix.csv", index=False)

    # --- PART C: FINAL PRODUCTION EXECUTION REFERENCE ---
    final_execution_prominence = 0.0015
    production_target_k = 10
    
    action_profile_registry = []
    kinematic_vectors = []
    segment_metadata_list = []
    total_proposed_segments = 0
    perfectly_pure_segments = 0
    matched_gt_boundaries = 0

    for ep_idx, stream in enumerate(in_memory_episodes):
        v_data = stream["conv"].to(DEVICE)
        a_data = stream["actions"]
        
        boundaries, proposed_cuts = extract_peaks_from_episode(v_data, final_execution_prominence)
        gt_intervals = atomic_vla_gt.get(ep_idx, [])
        gt_transition_frames = [step[1] for step in gt_intervals[:-1]]

        for gt_val in gt_transition_frames:
            if any(abs(proposed_cut - gt_val) <= 4 for proposed_cut in proposed_cuts):
                matched_gt_boundaries += 1

        for s_idx in range(len(boundaries) - 1):
            start, end = boundaries[s_idx], boundaries[s_idx+1]
            total_proposed_segments += 1

            is_pure = False
            for gt_start, gt_end, _ in gt_intervals:
                if start >= gt_start and end <= gt_end + 2:
                    is_pure = True
                    break
            if is_pure:
                perfectly_pure_segments += 1

            kinematic_signature = a_data[start:end].mean(dim=0)
            norm_kinematic = kinematic_signature / (kinematic_signature.norm() + 1e-8)

            kinematic_vectors.append(norm_kinematic.numpy())
            action_profile_registry.append({
                "uid": f"Ep{ep_idx}_Seg{s_idx}",
                "window_label": f"Ep{ep_idx}_Seg{s_idx} [{start}-{end}]"
            })
            segment_metadata_list.append({
                "episode_id": str(ep_idx), "start": int(start), "end": int(end)
            })

    X = np.array(kinematic_vectors)
    cluster_labels, _ = run_agglomerative_clustering(X, production_target_k)

    grouped_buckets = {i: [] for i in range(production_target_k)}
    for idx, label in enumerate(cluster_labels):
        grouped_buckets[label].append(action_profile_registry[idx]["window_label"])

    manifest_rows = []
    json_vocabulary_groups = {}
    for skill_id in sorted(grouped_buckets.keys()):
        cluster_members = grouped_buckets[skill_id]
        manifest_rows.append({
            "Uniform Global Skill ID": f"GLOBAL_SUB_SKILL_{skill_id}",
            "Total Discovered Chunks Included": len(cluster_members),
            "Unified Member Segments List": ", ".join(cluster_members)
        })
        json_vocabulary_groups[f"GLOBAL_SUB_SKILL_{skill_id}"] = cluster_members

    df_manifest = pd.DataFrame(manifest_rows).sort_values(by="Total Discovered Chunks Included", ascending=False)
    df_manifest.to_csv("global_sub_action_groups.csv", index=False)
       
    with open("global_sub_action_groups.json", "w") as jf:
        json.dump(json_vocabulary_groups, jf, indent=4)

    json_export_structure = {}
    for idx, cluster_id in enumerate(cluster_labels):
        meta = segment_metadata_list[idx]
        ep_key = meta["episode_id"]
        if ep_key not in json_export_structure:
            json_export_structure[ep_key] = []
        json_export_structure[ep_key].append({
            "start": meta["start"], "end": meta["end"], "cluster_id": int(cluster_id)
        })
        
    with open("unsupervised_sub_skills.json", "w") as json_file:
        json.dump(json_export_structure, json_file, indent=4)

    # --- REPORT GENERATION ---
    execution_time_seconds = time.time() - green_analytics["start_time"]
    hours = execution_time_seconds / 3600.0
    estimated_kwh = hours * (GPU_TDP_WATTS / 1000.0) * PUE_FACTOR
    estimated_carbon_kg = estimated_kwh * CARBON_INTENSITY

    purity_rate = (perfectly_pure_segments / total_proposed_segments) * 100 if total_proposed_segments > 0 else 0.0
    boundary_recall = (matched_gt_boundaries / total_gt_boundaries) * 100 if total_gt_boundaries > 0 else 100.0

    report_content = [
        "\n" + "="*165,
        "                                           COMPLETE VLA COMPRESSION THREE-DIMENSIONAL ABLATION MATRIX",
        "="*165,
        df_sweep.to_markdown(index=False),
        "="*165,
        f"\nUNSUPERVISED PRODUCTION REPORT (SPATIAL CONV | PROMINENCE={final_execution_prominence} | TARGET K={production_target_k})",
        "="*165,
        df_manifest.to_markdown(index=False),
        "="*165,
        "\nATOMIC VLA GROUND-TRUTH ALIGNMENT REPORT",
        f" Total Proposed Micro-Segments : {total_proposed_segments} Chunks",
        f" Sub-Atomic Semantic Purity    : {purity_rate:.2f}% ({perfectly_pure_segments}/{total_proposed_segments} nested inside GT)",
        f" Macro Boundary Recall Rate    : {boundary_recall:.2f}% ({matched_gt_boundaries}/{total_gt_boundaries} macro points matched)",
        "="*60,
        "\nGREEN AI METRICS REPORT",
        f" Total Compute Operations   : {green_analytics['total_flops']:.4e} FLOPs",
        f" Peak CUDA Memory Usage     : {green_analytics['peak_memory_bytes'] / (1024**3):.3f} GB",
        f" Runtime Execution Duration : {execution_time_seconds:.2f} seconds",
        f" Estimated Energy Usage     : {estimated_kwh:.5f} kWh",
        f" Estimated Carbon Footprint  : {estimated_carbon_kg:.5f} kg CO2e",
        "="*60 + "\n"
    ]

    final_report_text = "\n".join(report_content)
    print(final_report_text)

    with open("vla_segmentation_report.txt", "w", encoding="utf-8") as f:
        f.write(final_report_text)

if __name__ == "__main__":
    run_hierarchical_segment_grouping()