"""Chroma DB module."""
import os
from typing import Mapping, Sequence

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from logger import log
from database.base_database import BaseVectorDB
from models.base_embedding_model import EmbeddingModel


class BaseChromaDB(BaseVectorDB):
    """Base Chroma DB."""

    def __init__(self, collection_name: str) -> None:
        """Initialize the Chroma DB."""
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(name=collection_name)

    def save_embedding(self, text_data: str | list[str], embeddings: list[Sequence[float]], ids: list[str]) \
            -> None:
        """
        Save embedding data to Chroma collection.

        text_data: List of documents' text data.
        embeddings: List of embeddings corresponding to the documents.
        """
        try:
            self.collection.add(documents=text_data, embeddings=embeddings, ids=ids)
            log.info("Embeddings saved successfully.")
        except Exception as e:
            log.error(f"An error occurred: {e}")
            raise e


class TempChromaDB(BaseChromaDB):
    """Temporary Chroma DB."""

    def __init__(self, collection_name, embedding_function: EmbeddingFunction) -> None:
        """Initialize the temporary Chroma DB."""
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(name=collection_name,
                                                               embedding_function=embedding_function)

    def add_to_db(self, documents: str | list[str],
                  metadatas: Mapping[str, str | int | float | bool] | list[Mapping[str, str | int | float | bool]]
                  | None, ids: str | list[str]) -> None:
        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)


class PersistentChromaDB(BaseChromaDB):
    """Persistent Chroma DB."""

    def __init__(self, collection_name: str, embedding_function: EmbeddingFunction, path: str) -> None:
        """Initialize the persistent Chroma DB."""
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
