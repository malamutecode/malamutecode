"""Base class for embedding models."""
from abc import ABC, abstractmethod
from typing import Callable

from torch import Tensor
from torch.cuda import is_available


class EmbeddingModel(ABC):
    """Base class for embedding models."""

    def __init__(self, model_path: str) -> None:
        """Initialize the model."""
        self.model_path = model_path
        self.device = "cuda" if is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model = self._load_model()  # TODO: add batch support

    @abstractmethod
    def _load_model(self) -> Callable:
        """Load the model."""
        pass

    @abstractmethod
    def encode(self, sentence: str) -> Tensor:
        """Encode an image."""
        pass
