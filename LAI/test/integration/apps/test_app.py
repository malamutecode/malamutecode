"""Unit test for LAI apps."""
import os

import pytest

from apps.app import AlphaLAI
from test import CACHE_DIR

EXAMPLE_SATEMENT_FILENAME_1 = 'example_statement_1.pdf'
cached_file_path = os.path.join(CACHE_DIR, EXAMPLE_SATEMENT_FILENAME_1)


@pytest.fixture
def app():
    return AlphaLAI()


def test_insert_and_query_db_data(app):
    loaded_file = app.load_file(cached_file_path)
    app.insert_data_to_db(loaded_file, 3)
    results = app.query_db('prowadził samochod')
    assert 'samochodem' in results['documents'][0][0]


def test_prompt_with_rag(app):
    loaded_file = app.load_file(cached_file_path)
    app.insert_data_to_db(loaded_file, 3)
    response = app.prompt_with_rag("Czy w samochodzie był ktoś pijany")
    print(response)
