"""Auto Tokenizer Module."""
from typing import Any

import torch
from transformers import AutoTokenizer, BatchEncoding

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
        device_map = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return AutoTokenizer.from_pretrained(self.model_name_or_path, device_map=device_map)

    def encode(self, sentence: str) -> Any:
        """Encode a sentence."""
        encoded_input = self.tokenizer.encode(sentence, return_tensors=self.return_tensors_type)
        if torch.cuda.is_available():
            return encoded_input.to("cuda")
        return encoded_input

    def decode(self, tokens) -> str:
        """Decode tokens."""
        return self.tokenizer.decode(tokens)

    def pack_prompt(self, input_text: str) -> str:
        dialogue_template = [
            {"role": "user",
             "content": input_text}
        ]

        # Apply the chat template
        prompt = self.tokenizer.apply_chat_template(conversation=dialogue_template,
                                               tokenize=False,
                                               add_generation_prompt=True)
        return prompt

    def batch_encode(self, templated_prompt: str) -> BatchEncoding:
        input_ids = self.tokenizer(templated_prompt, return_tensors="pt").to("cuda")
        if torch.cuda.is_available():
            input_ids.to("cuda")
        return input_ids