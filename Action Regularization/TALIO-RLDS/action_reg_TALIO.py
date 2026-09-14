# Action Regularization with TALIO
## Add this class directly to datasets.py to wrap the standard RLDSDataset stream cleanly
class TALIOCompressedDataset(IterableDataset):
    def __init__(self, base_rlds_dataset: RLDSDataset, talio_json_path: str):
        self.base_dataset = base_rlds_dataset

        # Load pre-computed boundaries
        with open(talio_json_path, "r") as f:
            self.sub_skills_registry = json.load(f)

        # Copy required attributes so finetune.py/collators see no difference
        self.dataset_statistics = base_rlds_dataset.dataset_statistics
        self.dataset_length = base_rlds_dataset.dataset_length

    def __iter__(self):
        # Access the raw numpy iterator from the underlying TFDS pipeline
        for rlds_batch in self.base_dataset.dataset.as_numpy_iterator():

            # Extract standard tracking parameters from RLDS batch metadata
            episode_id = rlds_batch.get("episode_metadata", {}).get("episode_id", [b""])[0].decode()
            frame_idx = int(rlds_batch.get("absolute_frame_id", [-1])[0])

            # Apply our compression logic directly to the raw action arrays if a match is found
            if episode_id in self.sub_skills_registry and frame_idx != -1:
                segments = self.sub_skills_registry[episode_id]
                for seg in segments:
                    if seg["start"] <= frame_idx < seg["end"]:
                        # 1. Locate the dynamic anchor index (e.g., the midpoint of the discovered sub-action)
                        anchor_idx = (seg["start"] + seg["end"]) // 2

                        # 2. Overwrite the current frame state with the visual anchor state if tracking a single keyframe,
                        # or aggregate actions across the segment window
                        raw_actions = rlds_batch["action"] # Shape: (window_size + future_action_window_size, ACTION_DIM)

                        # Apply our temporal compression calculation directly to the raw action array
                        # before it hits tokenization
                        mean_action = np.mean(raw_actions, axis=0)
                        rlds_batch["action"] = np.tile(mean_action, (len(raw_actions), 1))
                        break

            # Pass the modified raw batch directly into the standard transform pipeline
            yield self.base_dataset.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

##  Then wrap the train_dataset of the RLDSDataset with TALIO action regularization class in finetune.py
# train_dataset = RLDSDataset(cfg.data_root_dir, cfg.dataset_name, batch_transform, ...)
# train_dataset = TALIOCompressedDataset(train_dataset, "datasets/unsupervised_sub_skills.json")