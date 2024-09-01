"""Base class for tokenizers."""
from abc import ABC, abstractmethod
from typing import Callable

from torch import Tensor


class Tokenizer(ABC):
    """Base class for tokenizers."""

    def __init__(self) -> None:
        """Initialize the tokenizer."""
        self.tokenizer = self._load_tokenizer()

    @abstractmethod
    def _load_tokenizer(self) -> Callable:
        """Load the tokenizer."""
        pass

    @abstractmethod
    def encode(self, sentence: str) -> Tensor:
        """Encode a sentence."""
        pass

    @abstractmethod
    def decode(self, tokens) -> str:
        """Decode tokens."""
        pass
