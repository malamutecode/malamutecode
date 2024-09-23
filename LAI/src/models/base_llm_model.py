"""Base class for LLM models."""
from abc import ABC, abstractmethod
from typing import Any

from transformers import BatchEncoding

from models.base_tokenizer import Tokenizer


class LLMModel(ABC):
    """Base class for LLM models."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        """Initialize the model."""
        self.tokenizer = tokenizer
        self.model = self._load_model()

    @abstractmethod
    def _load_model(self) -> Any:
        """Load the model."""
        pass

    @abstractmethod
    def generate(self, prompt: str, max_length: int = 50) -> str:
        """Generate text."""
        pass

    @abstractmethod
    def generate_from_packed_prompt(self, packed_prompt: BatchEncoding):
        pass
