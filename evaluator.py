from tqdm import tqdm
from similarity import SimilarityFunction
from clip import CLIPModelWrapper
from database import EmbeddingDatabase
from typing import List, Tuple
from ranking import Ranking

# ===== CLIP BENCHMARKER CLASS =====
class CLIPBenchmarker:
    def __init__(self, config, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_name = model_name
        self.embedding_model = CLIPModelWrapper(config, model_name)
        self.databases = {}
        self.results = {}
        self.config = config

    def load_dataset(self, dataset_name: str, dataset_class):
        """Load and process a dataset."""
        print(f"Loading {dataset_name} dataset...")

        dataset = dataset_class(config=self.config)
        database = EmbeddingDatabase(dataset, self.embedding_model, config=self.config)
        database.chunk_and_crop_and_embed()
        
        self.databases[dataset_name] = database
        print(f"Loaded {dataset_name} with {len(database.metadata)} samples")
    
    def run_benchmark(self, dataset_name: str, similarity_classes: List[SimilarityFunction]):
        """Run benchmark on a dataset with different similarity functions."""
        print(f"\nRunning benchmark on {dataset_name}...")
        
        database = self.databases[dataset_name]        
        dataset_results = {}
        
        for sim_class in similarity_classes:
            print(f"Testing {sim_class.name} similarity...")

            similarities = sim_class.compute_similarity(database.image_embeddings, database.text_embeddings)
            ranker = Ranking(similarities, database.texts_image_index, 
                             config=self.config)
            metrics = ranker.get_results()
            dataset_results[sim_class.name] = metrics
            
            print(f"{sim_class.name} Results:")
            ranker.display_result()

        self.results[dataset_name] = dataset_results
    
    def print_summary(self):
        """Print summary of all benchmark results."""
        print("\n" + "="*50)
        print("BENCHMARK SUMMARY")
        print("="*50)
        
        for dataset_name, dataset_results in self.results.items():
            print(f"\n{dataset_name.upper()}:")
            print("-" * 30)
            
            for sim_name, metrics in dataset_results.items():
                print(f"{sim_name}:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.4f}")
                print()
