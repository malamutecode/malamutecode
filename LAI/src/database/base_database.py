"""Module with base class of vector databases."""
from abc import ABC, abstractmethod

from torch import Tensor


class BaseVectorDB(ABC):
    """Base class for storing vector data."""

    @abstractmethod
    def save_embedding(self, text_data: str, embeddings: Tensor, ids: list[str]) -> None:
        pass
