import torch
import numpy as np
import pandas as pd
from typing import Dict, List
from clip_benchmark.metrics.zeroshot_retrieval import recall_at_k, batchify

class Ranking:
    """
    Computes CLIP-style retrieval metrics from similarity matrices (image-to-text and text-to-image).
    """

    def __init__(
        self,
        sim_dict: Dict[str, np.ndarray],
        texts_image_index: torch.Tensor,
        device: str = "cuda",
        batch_size: int = 64,
        recall_k_list: List[int] = [1, 5, 10],
    ):
        self.sim_dict = sim_dict
        self.device = device
        self.batch_size = batch_size
        self.recall_k_list = recall_k_list
        self.texts_image_index = texts_image_index
        self.metrics_df = self._compute_all_metrics()

    def _compute_all_metrics(self) -> pd.DataFrame:
        """
        Compute retrieval metrics for all similarity matrices.

        Returns:
            pd.DataFrame: Merged metrics with i2t and t2i results in a single row per similarity matrix.
        """
        all_metrics = []

        for name, sim_matrix in self.sim_dict.items():
            scores = torch.tensor(sim_matrix, dtype=torch.float32, device=self.device)

            # Ensure shape: [num_texts, num_images]
            expected_shape = (
                len(self.texts_image_index),
                len(self.index_mappings["flattend_image_idx"]),
            )
            if scores.shape != expected_shape:
                scores = scores.T
            assert (
                scores.shape == expected_shape
            ), f"[{name}] Invalid shape {scores.shape}, expected {expected_shape}"

            # Build positive pair matrix
            positive_pairs = torch.zeros_like(scores, dtype=torch.bool)
            positive_pairs[torch.arange(len(scores)), self.texts_image_index] = True

            i2t_metrics = {}
            t2i_metrics = {}

            for k in self.recall_k_list:
                # Image-to-Text (rows are images)
                i2t_recall = batchify(
                    recall_at_k,
                    scores.T,
                    positive_pairs.T,
                    self.batch_size,
                    self.device,
                    k=k,
                )
                i2t_metrics[f"i2t-R@{k}"] = round(
                    (i2t_recall > 0).float().mean().item() * 100, 2
                )

                # Text-to-Image (rows are texts)
                t2i_recall = batchify(
                    recall_at_k,
                    scores,
                    positive_pairs,
                    self.batch_size,
                    self.device,
                    k=k,
                )
                t2i_metrics[f"t2i-R@{k}"] = round(
                    (t2i_recall > 0).float().mean().item() * 100, 2
                )

            # Compute averages
            i2t_metrics["i2t-RAVG"] = round(
                np.mean([i2t_metrics[f"i2t-R@{k}"] for k in self.recall_k_list]), 2
            )
            t2i_metrics["t2i-RAVG"] = round(
                np.mean([t2i_metrics[f"t2i-R@{k}"] for k in self.recall_k_list]), 2
            )

            # Combine and save
            all_metrics.append(
                {
                    "Sim Type": name,
                    **i2t_metrics,
                    **t2i_metrics,
                    "mR": round(
                        (i2t_metrics["i2t-RAVG"] + t2i_metrics["t2i-RAVG"]) / 2, 2
                    ),
                }
            )

        # Define proper column order
        col_order = (
            ["Sim Type"]
            + [f"i2t-R@{k}" for k in self.recall_k_list]
            + ["i2t-RAVG"]
            + [f"t2i-R@{k}" for k in self.recall_k_list]
            + ["t2i-RAVG"]
            + ["mR"]
        )

        return pd.DataFrame(all_metrics)[col_order]

    def get_results(self) -> pd.DataFrame:
        """
        Returns:
            pd.DataFrame: Unified retrieval metric table.
        """
        return self.metrics_df

    def display_result(self):
        """Displays the metrics table."""
        print("\nRetrieval Metrics (CLIP-style)")
        print(self.metrics_df)