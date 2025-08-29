import torch

# ===== SIMILARITY CLASSES =====
class SimilarityFunction:
    def __init__(self, name: str):
        self.name = name

    def compute_similarity(self, text_embeddings: torch.Tensor, image_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute similarity between sets of text and image embeddings."""
        raise NotImplementedError

class MeanMeanSimilarity(SimilarityFunction):
    def __init__(self):
        super().__init__("mean_mean")
    
    def compute_similarity(self, text_embeddings: torch.Tensor, image_embeddings: torch.Tensor) -> torch.Tensor:
        """Average of all pairwise similarities."""
        # text_embeddings: [M, B, D], image_embeddings: [N, C, D]
        similarities = torch.matmul(text_embeddings, image_embeddings.transpose(1, 2))  # [M, B, C]
        return similarities.mean(dim=[1, 2])  # [M]

class MaxMeanSimilarity(SimilarityFunction):
    def __init__(self):
        super().__init__("max_mean")

    def compute_similarity(self, text_embeddings: torch.Tensor, image_embeddings: torch.Tensor) -> torch.Tensor:
        """Each text chunk matches its best image patch, then average."""
        # text_embeddings: [M, B, D], image_embeddings: [N, C, D]
        similarities = torch.matmul(text_embeddings, image_embeddings.transpose(1, 2))  # [M, B, C]
        max_similarities = similarities.max(dim=2)[0]  # [M, B]
        return max_similarities.mean(dim=1)