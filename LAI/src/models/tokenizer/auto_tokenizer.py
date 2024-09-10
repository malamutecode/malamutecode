"""Auto Tokenizer Module."""
from typing import Any

import torch
from transformers import AutoTokenizer

from models.base_tokenizer import Tokenizer


class AutoTokenizerModel(Tokenizer):
    """Auto Tokenizer Module."""

    def __init__(self, model_name_or_path: str) -> None:
        """
        Initialize the model.
        :param model_name_or_path: Model name on Hugging Face or path to model.
        """
        self.model_name_or_path = model_name_or_path
        self.return_tensors_type = "pt"
        super().__init__()

    def _load_tokenizer(self) -> AutoTokenizer:
        """Load the tokenizer."""
        return AutoTokenizer.from_pretrained(self.model_name_or_path)

    def encode(self, sentence: str) -> Any:
        """Encode a sentence."""
        encoded_input = self.tokenizer.encode(sentence, return_tensors=self.return_tensors_type)
        if torch.cuda.is_available():
            return encoded_input.to("cuda")
        return encoded_input

    def decode(self, tokens) -> str:
        """Decode tokens."""
        return self.tokenizer.decode(tokens)
