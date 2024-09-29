"""Test for Chroma DB"""
import pytest

from database.chroma_db import BaseChromaDB, SentenceEmbeddingFunction, TempChromaDB
from models.embedding_model.sentence_transformer_model import SentenceTransformerModel


@pytest.fixture
def chroma_db():
    return BaseChromaDB('test')


def test_save_embeddings(chroma_db):
    documents = ["This is a document about AI.", "This is another document about machine learning."]
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]  # Example embeddings
    ids = ['chunk_1', "chunk_2"]
    chroma_db.save_embedding(documents, embeddings, ids)


def test_save_embedding_with_custom_embedding_fn():
    db = TempChromaDB('test', SentenceEmbeddingFunction(SentenceTransformerModel("all-mpnet-base-v2")))
    db.add_to_db(
        documents=["This is a document", "This is another document"],
        metadatas=[{"source": "my_source"}, {"source": "my_source"}],
        ids=["id1", "id2"]
    )
    results = db.collection.query(
        query_texts=["This is a query document"],
        n_results=2
    )
    assert len(results['distances'][0]) == 2
