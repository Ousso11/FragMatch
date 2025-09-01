import torch
from abc import ABC, abstractmethod
# ===== SIMILARITY CLASSES =====
class SimilarityFunction(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute_similarity(self, text_embeddings: torch.Tensor, image_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute similarity between sets of text and image embeddings."""
        raise NotImplementedError

class MeanMeanSimilarity(SimilarityFunction):
    def __init__(self):
        super().__init__("mean_mean")
    
    def compute_similarity(self, text_embeddings: torch.Tensor, image_embeddings: torch.Tensor) -> torch.Tensor:
        """Average of all pairwise similarities."""
        # text_embeddings: [M, B, D], image_embeddings: [N, C, D]
        similarities = torch.einsum("mbd,ncd->mnbc", text_embeddings, image_embeddings) # [M, N, B]
        return similarities.mean(dim=[2, 3])  # [M, N]

class MaxMeanSimilarity(SimilarityFunction):
    def __init__(self):
        super().__init__("max_mean")

    def compute_similarity(self, text_embeddings: torch.Tensor, image_embeddings: torch.Tensor) -> torch.Tensor:
        """Each text chunk matches its best image patch, then average."""
        # text_embeddings: [M, B, D], image_embeddings: [N, C, D]
        similarities = torch.einsum("mbd,ncd->mnbc", text_embeddings, image_embeddings)
        max_similarities = similarities.max(dim=-1).values  # [M, N, B]
        return max_similarities.mean(dim=-1)  # [M, N]