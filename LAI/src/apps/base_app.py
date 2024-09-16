"""Module with base pipeline."""
from abc import ABC, abstractmethod

from chromadb import EmbeddingFunction

from data.file_loaders import BaseDataProcessor
from data.text_preprocessing import BaseTextProcessor
from database.base_database import BaseVectorDB
from models.base_embedding_model import EmbeddingModel
from models.base_llm_model import LLMModel
from models.base_tokenizer import Tokenizer


class BaseApp(ABC):
    """Struct for RAG pipelines."""

    @abstractmethod
    def _get_database(self) -> BaseVectorDB:
        pass

    @abstractmethod
    def _get_embedding_model(self) -> EmbeddingModel:
        pass

    @abstractmethod
    def _get_embedding_function(self) -> EmbeddingFunction:
        pass

    @abstractmethod
    def _get_llm_model(self) -> LLMModel:
        pass

    @abstractmethod
    def _get_tokenizer(self) -> Tokenizer:
        pass

    @abstractmethod
    def _get_text_preprocessor(self) -> BaseTextProcessor:
        pass

    @abstractmethod
    def _get_input_data_loader(self) -> BaseDataProcessor:
        pass
