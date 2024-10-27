"""Unit test for LAI apps."""
import os

from apps.app import AlphaLAI
from logger import log
from test import CACHE_DIR

EXAMPLE_SATEMENT_FILENAME_1 = 'example_statement_1.pdf'
cached_file_path = os.path.join(CACHE_DIR, EXAMPLE_SATEMENT_FILENAME_1)


def get_app():
    return AlphaLAI()


def insert_and_query_db_data_test(app: AlphaLAI) -> None:
    loaded_file = app.load_file(cached_file_path)
    app.insert_data_to_db(loaded_file, 3)
    results = app.query_db('prowadził samochod')
    assert 'samochodem' in results['documents'][0][0]


def model_in_app_test(app):
    output = app.model.generate("Ile jest 2 plus 2?")
    print(output)


def prompt_with_rag_test(app: AlphaLAI) -> None:
    loaded_file = app.load_file(cached_file_path)
    app.insert_data_to_db(loaded_file, 3)
    response = app.prompt_with_rag("Czy w samochodzie był ktoś pijany")
    log.info(response)


if __name__ == '__main__':
    insert_and_query_db_data_test(get_app())
    prompt_with_rag_test(get_app())
    model_in_app_test(get_app())
