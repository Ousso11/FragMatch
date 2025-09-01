import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
from clip import CLIPModelWrapper
from dataset_loader import custom_collate_fn
from fragmentation import TextSplitter, ImageCropper
from torch.nn.utils.rnn import pad_sequence

# ===== DATABASE CLASS =====
class EmbeddingDatabase:
    def __init__(self, dataset: Dataset, embedding_model: CLIPModelWrapper, config: dict):
        self.dataset = dataset
        self.embedding_model = embedding_model
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = config.batch_size

        # Initialize processors
        self.text_splitter = TextSplitter(config=config)
        self.image_cropper = ImageCropper(config=config)

        # Storage for embeddings
        self.image_embeddings = []  # List of tensors (one per sample, multiple crops each)
        self.text_embeddings = []   # List of tensors (one per sample, multiple chunks each)
        self.metadata = []          # List of dicts with sample info
        
        self._cache_file = self.cache_dir / f"embeddings_{type(dataset).__name__}_{len(dataset)}.pkl"
    
    def chunk_and_crop_and_embed(self, force_recompute: bool = False):
        """Process dataset: chunk text, crop images, embed, and cache."""
        
        if not force_recompute and self._cache_file.exists():
            print(f"Loading cached embeddings from {self._cache_file}")
            self._load_cache()
            return
        
        print("Processing dataset: chunking text, cropping images, and embedding...")
        
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, collate_fn=custom_collate_fn)
        self.texts_image_index = []
        self.image_embeddings = []
        self.text_embeddings = []
        self.len_image_crops = []
        self.len_text_chunks = []
        self.metadata = []

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            crops = self.image_cropper.crop_images(batch["images"])
            chunks = self.text_splitter.split_text(batch["captions"])
            flattend_chunks = [c for sublist in chunks for c in sublist]
            embeddings = self.embedding_model.batch_encode_image_text_pairs(crops, flattend_chunks)
            
            self.image_embeddings.extend(embeddings["image_embedding"])
            self.text_embeddings.extend(embeddings["text_embedding"])
            self.len_image_crops.extend(embeddings["image_patch_sizes"])
            self.len_text_chunks.extend(embeddings["text_patch_sizes"])
            indices = [i + batch_idx*self.batch_size for i, size in enumerate(embeddings["image_patch_sizes"]) for _ in range(size)]
            self.texts_image_index.extend(indices)
            self.metadata.extend([{"image_id": img_id, "caption": cap, "text_id": text_id} 
                                  for img_id, cap, text_id in zip(batch["image_ids"], chunks, batch["caption_ids"])])
            
        self.texts_image_index = torch.tensor(self.texts_image_index, dtype=torch.long)
        self.len_image_crops = torch.tensor(self.len_image_crops, dtype=torch.long)
        self.len_text_chunks = torch.tensor(self.len_text_chunks, dtype=torch.long)
        
                
        self.image_embeddings = pad_sequence(self.image_embeddings, batch_first=True)
        self.text_embeddings = pad_sequence(self.text_embeddings, batch_first=True)
        
       
        print(f"Image embeddings shape: {self.image_embeddings.shape}")
        print(f"Text embeddings shape: {self.text_embeddings.shape}")
        print(f"Processed {len(self.dataset)} samples")
        self._save_cache()
    
    def _save_cache(self):
        """Save embeddings and metadata to cache."""
        cache_data = {
            "image_embeddings": self.image_embeddings,
            "text_embeddings": self.text_embeddings,
            "len_image_crops": self.len_image_crops,
            "len_text_chunks": self.len_text_chunks,
            "texts_image_index": self.texts_image_index,
            "metadata": self.metadata
        }
        
        with open(self._cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"Cached embeddings saved to {self._cache_file}")
    
    def _load_cache(self):
        """Load embeddings and metadata from cache."""
        with open(self._cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        self.image_embeddings = cache_data["image_embeddings"]
        self.text_embeddings = cache_data["text_embeddings"]
        self.len_image_crops = cache_data["len_image_crops"]
        self.len_text_chunks = cache_data["len_text_chunks"]
        self.metadata = cache_data["metadata"]
        self.texts_image_index = cache_data["texts_image_index"]
        print(f"Loaded {len(self.metadata)} samples from cache")