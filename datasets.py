from PIL import Image
from typing import List, Dict
from torch.utils.data import Dataset
from pathlib import Path
import json

# ===== DATASET CLASSES =====
class FlickrDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train", download: bool = True):
        self.data_dir = Path(data_dir)
        self.split = split
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        if download and not self._check_data_exists():
            self._download_data()
        
        self.data = self._load_data()
    
    def _check_data_exists(self) -> bool:
        """Check if dataset files exist."""
        expected_files = [
            self.data_dir / f"flickr30k_{self.split}.json",
            self.data_dir / "flickr30k_images"
        ]
        return all(f.exists() for f in expected_files)
    
    def _download_data(self):
        """Download Flickr30k dataset (mock implementation)."""
        print(f"Downloading Flickr30k {self.split} split...")
        # This is a mock implementation - in reality you'd download from official sources
        
        # Create mock data for demonstration
        mock_data = []
        for i in range(100):  # Create 100 mock samples
            mock_data.append({
                "image_id": f"flickr_image_{i:04d}.jpg",
                "caption": f"A sample caption number {i} describing various objects and scenes in the image with multiple details and descriptive elements.",
                "image_path": f"flickr30k_images/flickr_image_{i:04d}.jpg"
            })
        
        # Save mock data
        with open(self.data_dir / f"flickr30k_{self.split}.json", 'w') as f:
            json.dump(mock_data, f, indent=2)
        
        # Create images directory
        img_dir = self.data_dir / "flickr30k_images"
        img_dir.mkdir(exist_ok=True)
        
        # Create mock images
        for i in range(100):
            # Create a simple colored image
            img = Image.new('RGB', (224, 224), color=(i*2 % 255, (i*3) % 255, (i*5) % 255))
            img.save(img_dir / f"flickr_image_{i:04d}.jpg")
        
        print("Mock dataset created successfully!")
    
    def _load_data(self) -> List[Dict]:
        """Load dataset from JSON file."""
        json_file = self.data_dir / f"flickr30k_{self.split}.json"
        with open(json_file, 'r') as f:
            return json.load(f)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        
        # Load image
        image_path = self.data_dir / item["image_path"]
        image = Image.open(image_path).convert('RGB')
        
        return {
            "image": image,
            "caption": item["caption"],
            "image_id": item["image_id"],
            "index": idx
        }

class COCODataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train", download: bool = True):
        self.data_dir = Path(data_dir)
        self.split = split
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        if download and not self._check_data_exists():
            self._download_data()
        
        self.data = self._load_data()
    
    def _check_data_exists(self) -> bool:
        expected_files = [
            self.data_dir / f"coco_{self.split}.json",
            self.data_dir / "coco_images"
        ]
        return all(f.exists() for f in expected_files)
    
    def _download_data(self):
        print(f"Downloading COCO {self.split} split...")
        
        # Create mock data
        mock_data = []
        for i in range(100):
            mock_data.append({
                "image_id": f"coco_image_{i:04d}.jpg",
                "caption": f"COCO style caption {i} with objects, people, and detailed scene description including colors and spatial relationships.",
                "image_path": f"coco_images/coco_image_{i:04d}.jpg"
            })
        
        with open(self.data_dir / f"coco_{self.split}.json", 'w') as f:
            json.dump(mock_data, f, indent=2)
        
        img_dir = self.data_dir / "coco_images"
        img_dir.mkdir(exist_ok=True)
        
        for i in range(100):
            img = Image.new('RGB', (224, 224), color=((i*7) % 255, (i*11) % 255, (i*13) % 255))
            img.save(img_dir / f"coco_image_{i:04d}.jpg")
        
        print("Mock COCO dataset created successfully!")
    
    def _load_data(self) -> List[Dict]:
        json_file = self.data_dir / f"coco_{self.split}.json"
        with open(json_file, 'r') as f:
            return json.load(f)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        
        image_path = self.data_dir / item["image_path"]
        image = Image.open(image_path).convert('RGB')
        
        return {
            "image": image,
            "caption": item["caption"],
            "image_id": item["image_id"],
            "index": idx
        }

# ===== COLLATE FUNCTION =====
def custom_collate_fn(batch):
    """Custom collate function for handling variable-sized data."""
    images = [item["image"] for item in batch]
    captions = [item["caption"] for item in batch]
    image_ids = [item["image_id"] for item in batch]
    indices = [item["index"] for item in batch]
    
    return {
        "images": images,
        "captions": captions,
        "image_ids": image_ids,
        "indices": indices
    }
