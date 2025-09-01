import torch
from typing import List, Tuple

from abc import ABC, abstractmethod
# ===== SIMILARITY CLASSES =====
class SimilarityFunction(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute_similarity(self, text_embeddings: torch.Tensor, image_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute similarity between sets of text and image embeddings."""
        raise NotImplementedError

    def normalize(self, text_embeddings: torch.Tensor, image_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize text and image embeddings."""
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        return text_embeddings, image_embeddings

class MeanMeanSimilarity(SimilarityFunction):
    def __init__(self):
        super().__init__("mean_mean")
    
    def compute_similarity(self, text_embeddings: torch.Tensor, image_embeddings: torch.Tensor) -> torch.Tensor:
        """Average of all pairwise similarities."""
        # text_embeddings: [M, B, D], image_embeddings: [N, C, D]
        # text_embeddings, image_embeddings = self.normalize(text_embeddings, image_embeddings)
        similarities = torch.einsum("mbd,ncd->mnbc", text_embeddings, image_embeddings) # [M, N, B]
        return similarities.mean(dim=[2, 3])  # [M, N]

class MaxMeanSimilarity(SimilarityFunction):
    def __init__(self):
        super().__init__("max_mean")

    def compute_similarity(self, text_embeddings: torch.Tensor, image_embeddings: torch.Tensor) -> torch.Tensor:
        """Each text chunk matches its best image patch, then average."""
        # text_embeddings: [M, B, D], image_embeddings: [N, C, D]
        # text_embeddings, image_embeddings = self.normalize(text_embeddings, image_embeddings)
        similarities = torch.einsum("mbd,ncd->mnbc", text_embeddings, image_embeddings)
        max_similarities = similarities.max(dim=-1).values  # [M, N, B]
        return max_similarities.mean(dim=-1)  # [M, N]

class GlobalCosineSimilarity(SimilarityFunction):
    def __init__(self):
        super().__init__("global_cosine")

    def compute_similarity(self, text_embeddings: torch.Tensor, image_embeddings: torch.Tensor) -> torch.Tensor:
        """Global cosine similarity."""
        text_embeddings = text_embeddings[:,0,:]
        image_embeddings = image_embeddings[:,0,:]
        text_embeddings, image_embeddings = self.normalize(text_embeddings, image_embeddings)
        print("text shape:", text_embeddings.shape)
        print("image shape:", image_embeddings.shape)
        return torch.einsum("md,nd->mn", text_embeddings, image_embeddings)
