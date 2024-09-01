"""Sentence Transformer Model."""
from sentence_transformers import SentenceTransformer
from torch import Tensor

from models.embedding_model.base_embedding_model import EmbeddingModel


class SentenceTransformerModel(EmbeddingModel):
    """Sentence Transformer Model."""

    def __init__(self, model_name_or_path: str) -> None:
        """Initialize the model."""
        super().__init__(model_name_or_path)

    def _load_model(self) -> SentenceTransformer:
        """Load the model."""
        return SentenceTransformer(self.model_name_or_path, device=self.device)

    def encode(self, sentence: str) -> Tensor:
        """Encode a sentence."""
        embedding = self.model.encode(sentence, convert_to_tensor=True)
        assert isinstance(embedding, Tensor)
        return embedding


if __name__ == "__main__":
    model = SentenceTransformerModel("all-mpnet-base-v2")
    embedding = model.encode("Hello, my dog is cute.")
    print(embedding)
