import os
from typing import Sequence
from venv import logger

from chromadb.api.types import Documents, Embeddings, EmbeddingFunction
import chromadb
from torch import Tensor

from LAI.src import CACHE_DIR
from database.base_database import BaseVectorDB
from models.base_embedding_model import EmbeddingModel


CHROMA_PATH = os.path.join(CACHE_DIR, 'chroma')


class BaseChromaDB(BaseVectorDB):

    def __init__(self, collection_name: str) -> None:
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(name=collection_name)

    def save_embedding(self, text_data: str | list[str], embeddings: Tensor | list[Sequence[float]], ids: list[str]) \
            -> None:
        """
        Save embedding data to Chroma collection.

        text_data: List of documents' text data.
        embeddings: List of embeddings corresponding to the documents.
        """
        try:
            self.collection.add(documents=text_data, embeddings=embeddings, ids=ids)
            logger.info("Embeddings saved successfully.")
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise e


class TempChromaDB(BaseChromaDB):
    """Temporary Chroma DB."""

    def __init__(self, collection_name, embedding_function: EmbeddingFunction) -> None:
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(name=collection_name,
                                                               embedding_function=embedding_function)

    def add_to_db(self, documents: str | list[str], metadatas: dict[str, str] | list[dict[str, str]],
                  ids: str |list[str]) -> None:
        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)


class PersistentChromaDB(BaseChromaDB):
    """Persistent Chroma DB."""

    def __init__(self, collection_name: str, embedding_function: EmbeddingFunction, path: str) -> None:
        self.db_path = path
        is_new_db = os.path.exists(path)
        self.chroma_client = chromadb.PersistentClient(path=path)
        if is_new_db:
            self.chroma_client.create_collection(name=collection_name, embedding_function=embedding_function)
        else:
            self.chroma_client.get_collection(name=collection_name, embedding_function=embedding_function)


class SentenceEmbeddingFunction(EmbeddingFunction[Documents]):
    """Embedding function for Chroma DB."""

    def __init__(self, embedding_model: EmbeddingModel):
        """Initialize the embedding function."""
        self.model = embedding_model

    def __call__(self, input: Documents) -> Embeddings:
        """Embed the input documents."""
        return [self.model.encode(sentence).tolist() for sentence in input]
