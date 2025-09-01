import torch
from dataclasses import dataclass
from evaluator import CLIPBenchmarker
from similarity import GlobalCosineSimilarity, MeanMeanSimilarity, MaxMeanSimilarity
from dataset_loader import FlickrDataset, COCODataset
import argparse
from typing import Tuple 
get_device = lambda : "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

DATASETS = {
    "flickr30k": FlickrDataset,
    # "coco": COCODataset
}
SIMILARITY_FUNCTIONS = {
    "cosine": GlobalCosineSimilarity(),
    "mean": MeanMeanSimilarity(),
    "max": MaxMeanSimilarity()
}

# Configuration
@dataclass
class Config:
    cache_dir: str = "./cache"
    data_dir: str = "./data"
    split: str = "test"
    data_length: int = None
    batch_size: int = 128
    max_text_length: int = 77
    image_size: int = 224
    chunk_overlap: int = 0
    max_chunks: int = 10
    include_full_image: bool = True
    include_full_text: bool = True
    grid_size: int = 3  # 3x3 grid for image cropping
    device: str = get_device()
    recall_k_list: Tuple[int] = (1, 5, 10)

def main(args):
    config = Config()
    config.batch_size = args.batch_size
    config.data_length = args.data_length
    # Initialize benchmarker
    benchmarker = CLIPBenchmarker(config=config, model_name="openai/clip-vit-base-patch32")

    data = DATASETS.keys() if args.data == "all" else [args.data]
    similarities = SIMILARITY_FUNCTIONS.values() if args.similarity == "all" else [SIMILARITY_FUNCTIONS[args.similarity]]

    for dataset in data:
        benchmarker.load_dataset(dataset, DATASETS[dataset])
        benchmarker.run_benchmark(dataset, similarities)
    
# ===== MAIN EXECUTION EXAMPLE =====
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Run CLIP Benchmarking")
    arg_parser.add_argument("--data", type=str, choices=list(DATASETS.keys()) + ["all"], 
                            default="all", help="Dataset to benchmark")
    arg_parser.add_argument("--similarity", type=str, choices=["mean", "max", "all"], 
                            default="all", help="Similarity function to use")
    arg_parser.add_argument("--batch_size", type=int, default=128, help="Batch size for processing")
    arg_parser.add_argument("--data_length", type=int, default=1000, help="Number of samples to use from the dataset")
    main_args = arg_parser.parse_args()
    main(main_args)
