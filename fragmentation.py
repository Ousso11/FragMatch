import torch
from PIL import Image
from typing import List, Union, Tuple
import spacy
import torchvision.transforms as T
import torch.nn.functional as F


# ===== TEXT SPLITTING CLASS =====
class TextSplitter:
    def __init__(self, config):
        self.max_chunk_length = config.max_chunks
        self.overlap = config.chunk_overlap
        # Load spacy for better text processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Using simple splitting.")
            self.nlp = None

    def split_text(self, text: Union[List[str], str]
                   ) -> List[List[str]]:
        """Split text into meaningful chunks using multiple strategies."""
        if isinstance(text, str):
            text = [text]
        if not text or all(not t.strip() for t in text):
            return [[""]]
        
        # Strategy 1: Sentence-based splitting with spaCy
        if self.nlp:
            chunks = [self._split_with_spacy(t) for t in text]
        else:
            chunks = [self._split_simple(t) for t in text]

        return chunks

    def _split_with_spacy(self, text: str) -> List[str]:
        """Split using spaCy for better linguistic understanding."""
        doc = self.nlp(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_tokens = len(sent_text.split())
            
            if current_length + sent_tokens <= self.max_chunk_length:
                current_chunk.append(sent_text)
                current_length += sent_tokens
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sent_text]
                current_length = sent_tokens
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        # If no good splits found, fallback to simple splitting
        if not chunks or (len(chunks) == 1 and len(chunks[0].split()) > self.max_chunk_length):
            return self._split_simple(text)
        
        return chunks
    
    def _split_simple(self, text: str) -> List[str]:
        """Simple word-based splitting as fallback."""
        words = text.split()
        if len(words) <= self.max_chunk_length:
            return [text]
        
        chunks = []
        for i in range(0, len(words), self.max_chunk_length - self.overlap):
            chunk_words = words[i:i + self.max_chunk_length]
            chunks.append(" ".join(chunk_words))
        
        return chunks

# ===== IMAGE CROPPING CLASS =====

class ImageCropper:
    def __init__(self, config, grid_size=2, include_full_image=True):
        self.grid_size = grid_size
        self.include_full_image = include_full_image
        self.image_size = config.image_size

        # Full transformation pipeline for both images and patches
        self.transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711)
            )
        ])

    def crop_images(self, images: Union[Image.Image, List[Image.Image]]
                    ) -> List[torch.Tensor]:
        """
        Crop a single image or a batch of images into grid patches.
        Ensures full image and crops are transformed to the same size.
        Returns HWC patches:
            - Single image: num_patches x H x W x C
            - Batch: B x num_patches x H x W x C
        """
        single_image = False
        if isinstance(images, Image.Image):
            images = [images]
            single_image = True

        B = len(images)

        # Process the original images to get both full images and original tensors for cropping
        transformed_images = torch.stack([self.transform(img.convert("RGB")) for img in images])
        
        _, C, H_orig, W_orig = transformed_images.shape
        crop_h = H_orig // self.grid_size
        crop_w = W_orig // self.grid_size

        # Use F.unfold on the entire batch of original tensors
        patches = F.unfold(transformed_images, kernel_size=(crop_h, crop_w), stride=(crop_h, crop_w))

        # Reshape the patches and merge the batch and patch dimensions for processing
        num_patches = self.grid_size * self.grid_size
        patches = patches.view(B, C, crop_h, crop_w, num_patches)
        patches = patches.permute(0, 4, 1, 2, 3) # B x num_patches x C x h x w
        patches = patches.reshape(B * num_patches, C, crop_h, crop_w)

        # Convert the batch of patch tensors back to PIL images and apply the full transform
        patched_list = [T.ToPILImage()(p) for p in patches]
        transformed_patches = torch.stack([self.transform(p) for p in patched_list])
        
        # Reshape back to B x num_patches x C x H x W
        transformed_patches = transformed_patches.view(B, num_patches, C, self.image_size, self.image_size)

        # Concatenate the full images with the transformed patches
        if self.include_full_image:
            full_images = transformed_images.unsqueeze(1)
            result = torch.cat([full_images, transformed_patches], dim=1)
        else:
            result = transformed_patches

        results_list = [res for res in result] if not single_image else [result[0]]
        return results_list