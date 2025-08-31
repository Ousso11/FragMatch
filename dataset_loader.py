import os
import json
from PIL import Image
from typing import List, Dict
from torch.utils.data import Dataset
from pathlib import Path
import torch
from datasets import load_dataset
from tqdm import tqdm


HUB = {
    "flickr30k": "nlphuji/flickr30k"
}

class COCODataset(Dataset):
   pass

# ===== DATASET CLASSES =====

class FlickrDataset(Dataset):
    """
    Flickr30k Dataset loader that downloads from Hugging Face
    and organizes data into train/val/test splits.
    """
    
    def __init__(self, 
                 data_dir: str, 
                 split: str = "test",
                 download: bool = True,
                 transform=None):
        """
        Initialize Flickr30k dataset.
        
        Args:
            data_dir: Root directory for dataset storage
            download: Whether to download the dataset if not present
            transform: Optional transform to apply to images
            split: Which split to use (train/val/test)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split

        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.data_dirs = {
            "train": self.data_dir / "train",
            "val": self.data_dir / "val",
            "test": self.data_dir / "test"
        }
        for data_dir in self.data_dirs.values():
            data_dir.mkdir(parents=True, exist_ok=True)
            os.makedirs(data_dir / "images", exist_ok=True)

        # Download if needed
        if download and not self._check_data_exists():
            self._download_data()
        
        # Load data
        self.data = self._load_data()
        
    def _check_data_exists(self) -> bool:
        """Check if dataset files exist."""
        json_path = os.path.join(self.data_dirs[self.split], "annotations.json")
        if not os.path.exists(json_path):
            return False
        with open(json_path, 'r') as f:
            annotations = json.load(f)
        image_paths = [item["image_path"] for item in annotations.values()]
       
        if any(not os.path.exists(os.path.join(self.data_dir, img_path)) for img_path in image_paths):
            return False

        return True
        
    def _download_data(self):
        """Download Flickr30k dataset from Hugging Face."""
        print(f"Downloading Flickr30k dataset...")
                
        # Load dataset from Hugging Face
        dataset = load_dataset(HUB["flickr30k"], split="test")
        
        # Prepare annotations dictionary
        annotations = {"train": {}, "val": {}, "test": {}}

        print(f"Processing {len(dataset)} samples for {self.split} split...")

        for idx, item in enumerate(tqdm(dataset, desc=f"Processing data")):
            # Extract image ID and filename
            filename = item['filename']
            split = item['split']
            
            # Save image
            image_path = os.path.join(self.data_dirs[split], "images", filename)
            if not os.path.exists(image_path):
                try:
                    # Get image from the dataset
                    image = item['image']
                    if image is not None:
                        image.save(image_path)
                except Exception as e:
                    print(f"Error saving image {filename}: {e}")
                    continue

            # Prepare annotation entry
            annotations[split][item['img_id']] = {
                "image_id": item['img_id'],
                "image_path": os.path.join(split, "images", filename),
                "filename": filename,
                "captions": item['caption'],  # List of 5 captions
                "caption_ids": item['sentids']  # List of caption IDs
            }

        for split in ["train", "val", "test"]:
            # Save annotations to JSON
            json_path = self.data_dirs[split] / "annotations.json"
            with open(json_path, 'w') as f:
                json.dump(annotations[split], f, indent=2)

        print(f"Downloaded and saved {len(annotations)} samples for {split} split")

    def _load_data(self) -> List[Dict]:
        """Load dataset from JSON file."""
        json_path = self.data_dirs[self.split] / "annotations.json"

        if not json_path.exists():
            raise FileNotFoundError(f"Annotations file not found at {json_path}. "
                                  f"Please run with download=True first.")
        
        with open(json_path, 'r') as f:
            annotations = json.load(f)
        
        # Convert to list format for indexing
        data = []
        for image_id, item in annotations.items():
            data.append(item)

        print(f"Loaded {len(data)} samples for {self.split} split")
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single item from the dataset.
        
        Returns:
            Dictionary containing:
                - image: PIL Image or transformed image
                - captions: List of 5 caption strings
                - caption_ids: List of 5 caption IDs
                - image_id: Image identifier
                - image_path: Relative path to image
                - index: Dataset index
        """
        item = self.data[idx]
        
        # Load image
        image_path = self.data_dir / item["image_path"]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Create a dummy image if loading fails
            image = Image.new('RGB', (224, 224), color='gray')
        
        # Apply transform if provided
        if self.transform:
            image = self.transform(image)
        
        return {
            "image": image,
            "captions": item["captions"],  # List of 5 captions
            "caption_ids": item["caption_ids"],  # List of 5 caption IDs
            "image_id": item["image_id"],
            "image_path": item["image_path"],
            "index": idx
        }


def custom_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for DataLoader.
    
    Args:
        batch: List of dictionaries from __getitem__
        
    Returns:
        Dictionary with batched data:
            - images: List of PIL images or tensor if transformed
            - captions: List of lists of captions (batch_size x 5)
            - caption_ids: List of lists of caption IDs
            - image_ids: List of image IDs
            - image_paths: List of image paths
            - indices: List of dataset indices
    """
    return {
        'images': [item['image'] for item in batch],
        'captions': [item['captions'] for item in batch],
        'caption_ids': [item['caption_ids'] for item in batch],
        'image_ids': [item['image_id'] for item in batch],
        'image_paths': [item['image_path'] for item in batch],
        'indices': [item['index'] for item in batch]
    }
