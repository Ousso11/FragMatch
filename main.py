import torch
from dataclasses import dataclass
from evaluator import CLIPBenchmarker
from similarity import MeanMeanSimilarity, MaxMeanSimilarity
from dataset_loader import FlickrDataset, COCODataset
import argparse
get_device = lambda : "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

DATASETS = {
    "flickr30k": FlickrDataset,
    "coco": COCODataset
}
SIMILARITY_FUNCTIONS = {
    "mean": MeanMeanSimilarity(),
    "max": MaxMeanSimilarity()
}

# Configuration
@dataclass
class Config:
    cache_dir: str = "./cache"
    data_dir: str = "./data"
    split: str = "test"
    batch_size: int = 32
    max_text_length: int = 77
    image_size: int = 224
    chunk_overlap: int = 0
    max_chunks: int = 10
    include_full_image: bool = True
    grid_size: int = 3  # 3x3 grid for image cropping
    device: str = get_device()

def main(args):
    config = Config()

    # Initialize benchmarker
    benchmarker = CLIPBenchmarker(config=config, model_name="openai/clip-vit-base-patch32")

    data = DATASETS.keys() if args.data == "all" else [args.data]
    similarities = SIMILARITY_FUNCTIONS.values() if args.similarity == "all" else [SIMILARITY_FUNCTIONS[args.similarity]]

    for dataset in data:
        benchmarker.load_dataset(dataset, DATASETS[dataset])
        benchmarker.run_benchmark(dataset, similarities)
        benchmarker.print_results()
    
# ===== MAIN EXECUTION EXAMPLE =====
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Run CLIP Benchmarking")
    arg_parser.add_argument("--data", type=str, choices=list(DATASETS.keys()) + ["all"], 
                            default="all", help="Dataset to benchmark")
    arg_parser.add_argument("--similarity", type=str, choices=["mean", "max", "all"], 
                            default="all", help="Similarity function to use")
    main_args = arg_parser.parse_args()
    main(main_args)
