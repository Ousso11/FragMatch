import torch
from typing import List, Tuple
from transformers import CLIPModel, CLIPProcessor
from torch.nn import functional as F

# ===== CLIP MODEL WRAPPER =====
class CLIPModelWrapper:
    def __init__(self, config, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_configs = {
            "openai/clip-vit-base-patch32": "openai/clip-vit-base-patch32",
            "openai/clip-vit-large-patch14": "openai/clip-vit-large-patch14",
            "laion/CLIP-ViT-B-32-laion2B-s34B-b79K": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
        }
        
        self.model_name = model_name
        self.config = config
        self.device = config.device
        self._load_model()
    
    def _load_model(self):
        """Load CLIP model and processor."""
        hf_model_name = self.model_configs.get(self.model_name, self.model_name)
        
        print(f"Loading CLIP model: {hf_model_name}")
        self.model = CLIPModel.from_pretrained(hf_model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(hf_model_name)
        self.model.eval()
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode single text to embedding."""
        with torch.no_grad():
            inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            text_features = self.model.get_text_features(**inputs)
            return F.normalize(text_features, p=2, dim=-1)
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode single image tensor to embedding."""
        with torch.no_grad():
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)
            image_features = self.model.get_image_features(pixel_values=image)
            return F.normalize(image_features, p=2, dim=-1)
    
    def batch_encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode batch of texts to embeddings."""
        if not texts:
            return torch.empty(0, self.model.config.projection_dim).to(self.device)
        
        with torch.no_grad():
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            text_features = self.model.get_text_features(**inputs)
            return F.normalize(text_features, p=2, dim=-1)
    
    def batch_encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode batch of image tensors to embeddings."""
        if images.size(0) == 0:
            return torch.empty(0, self.model.config.projection_dim).to(self.device)
        
        with torch.no_grad():
            images = images.to(self.device)
            image_features = self.model.get_image_features(pixel_values=images)
            return F.normalize(image_features, p=2, dim=-1)
    
  
    
    def batch_encode_image_text_pairs(self, 
                                      image_batch: List[torch.Tensor], 
                                      text_batch: List[List[str]]
                                      ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Encode multiple batches of image-text pairs.
        image_batches: list of B tensors, each tensor of shape [#crops, H, W, C]
        text_batches: list of B lists, each list contains #chunks
        """
        
        # Get the number of crops/chunks per batch
        image_patch_sizes = [len(c) for c in image_batch]
        text_patch_sizes = [len(c) for c in text_batch]
        
        # Flatten the input lists and tensors for batch encoding
        # Use torch.cat for images to combine tensors efficiently
        flattend_images = torch.cat(image_batch, dim=0)
        # Use a list comprehension to flatten the list of text lists
        flattend_texts = [text for sublist in text_batch for text in sublist]
        
        # Encode all flattened images and texts in a single pass
        all_image_embeddings = self.batch_encode_image(flattend_images)
        all_text_embeddings = self.batch_encode_text(flattend_texts)
        
        # Reshape the embeddings back to the original batch structure
        # Use torch.split for both images and text for consistency and efficiency
        image_embeddings_list = list(torch.split(all_image_embeddings, image_patch_sizes))
        text_embeddings_list = list(torch.split(all_text_embeddings, text_patch_sizes))
            
        return {
            "image_embedding": image_embeddings_list,
            "text_embedding": text_embeddings_list,
            "image_patch_sizes": image_patch_sizes,
            "text_patch_sizes": text_patch_sizes
        }