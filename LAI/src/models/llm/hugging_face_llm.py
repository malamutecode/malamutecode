"""Hugging Face Language Model class."""
from typing import Any

import huggingface_hub
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available

from models.base_llm_model import LLMModel
from models.tokenizer.auto_tokenizer import AutoTokenizerModel

# TODO: Add huggingface_hub auto login function


class HuggingFaceLLM(LLMModel):
    """Hugging Face Language Model class."""

    def __init__(self, model_name_or_path: str, model_dtype: torch.dtype,
                 quantization_config: BitsAndBytesConfig | None,
                 *args: Any, **kwargs: Any) -> None:
        """
        Initialize the model.
        :param model_name_or_path: HuggingFace model name or path to model.
        :param model_dtype: Model data type.
        :param quantization_config: Quantization configuration.
        """
        self.model_name_or_path = model_name_or_path
        self.model_dtype = model_dtype
        self.quantization_config = quantization_config
        self.attention_implementation = self._get_attention_implementation()
        super().__init__(*args, **kwargs)

    @staticmethod
    def _get_attention_implementation() -> str:
        """Get flash_attention_2 if available else use 'sdpa'."""
        if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
            return "flash_attention_2"
        return "sdpa"

    def _load_model(self) -> AutoModelForCausalLM:
        return AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model_name_or_path, torch_dtype=self.model_dtype,
            quantization_config=self.quantization_config, attn_implementation=self.attention_implementation)

    def generate(self, prompt: str, max_length: int = 250) -> str:
        """Generate text."""
        input_ids = self.tokenizer.encode(prompt)
        output = self.model.generate(input_ids, max_length=max_length)
        return self.tokenizer.decode(output[0])


if __name__ == "__main__":
    huggingface_hub.login(token='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')

    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    tokenizer = AutoTokenizerModel("google/gemma-2b-it")
    model = HuggingFaceLLM("google/gemma-2b-it", torch.float16, bnb_config, tokenizer=tokenizer)
    generated_text = model.generate("Hello, why 2+2 is 4?")
    print(generated_text)
