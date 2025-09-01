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
        sim_matrix: torch.Tensor,
        texts_image_index: torch.Tensor,
        config: dict,
    ):
        self.sim_matrix = sim_matrix
        self.device = config.device
        self.batch_size = config.batch_size
        self.recall_k_list = config.recall_k_list
        self.texts_image_index = texts_image_index
        self.metrics_df = self._compute_all_metrics()

    def _compute_all_metrics(self) -> pd.DataFrame:
        """
        Compute retrieval metrics for all similarity matrices.

        Returns:
            pd.DataFrame: Merged metrics with i2t and t2i results in a single row per similarity matrix.
        """
        all_metrics = []

        # scores = torch.tensor(self.sim_matrix, dtype=torch.float32, device=self.device)

        # Ensure shape: [num_texts, num_images]
        expected_shape = (
            len(self.texts_image_index),
            len(self.texts_image_index.unique()),
        )
        if self.sim_matrix.shape != expected_shape:
            self.sim_matrix = self.sim_matrix.T
        assert (
            self.sim_matrix.shape == expected_shape
        ), f"Invalid shape {self.sim_matrix.shape}, expected {expected_shape}"

        # Build positive pair matrix
        positive_pairs = torch.zeros_like(self.sim_matrix, dtype=torch.bool)
        positive_pairs[torch.arange(len(self.sim_matrix)), self.texts_image_index] = True

        i2t_metrics = {}
        t2i_metrics = {}

        for k in self.recall_k_list:
            # Image-to-Text (rows are images)
            i2t_recall = batchify(
                recall_at_k,
                self.sim_matrix.T,
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
                self.sim_matrix,
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
                **i2t_metrics,
                **t2i_metrics,
                "mR": round(
                    (i2t_metrics["i2t-RAVG"] + t2i_metrics["t2i-RAVG"]) / 2, 2
                ),
            }
        )

        # Define proper column order
        col_order = (
             [f"i2t-R@{k}" for k in self.recall_k_list]
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