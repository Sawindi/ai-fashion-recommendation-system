"""
FashionCLIP embedding + similarity module.

Load FashionCLIP model
Generate embedding vectors for cropped garment images
Compute cosine similarity between garments
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


FASHIONCLIP_ID = "patrickjohncyh/fashion-clip"


class FashionCLIPEmbedder:
    def __init__(self, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading FashionCLIP on {self.device}...")
        self.model = CLIPModel.from_pretrained(FASHIONCLIP_ID).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(FASHIONCLIP_ID)

        self.model.eval()

    # Image -> Embeddings
   
    def get_image_embedding(self, image_path: str | Path) -> np.ndarray:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            # Run vision encoder
            outputs = self.model.vision_model(pixel_values=inputs["pixel_values"])
            # pooled_output shape: [1, hidden]
            pooled = outputs.pooler_output

            # Project to CLIP embedding space
            image_features = self.model.visual_projection(pooled)

        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy().flatten()


    # Similarity
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)

        return float(np.dot(vec1, vec2))

   
    # Batch similarity
    @staticmethod
    def similarity_matrix(embeddings: List[np.ndarray]) -> np.ndarray:
        matrix = np.zeros((len(embeddings), len(embeddings)))

        for i in range(len(embeddings)):
            for j in range(len(embeddings)):
                matrix[i, j] = FashionCLIPEmbedder.cosine_similarity(
                    embeddings[i], embeddings[j]
                )

        return matrix
