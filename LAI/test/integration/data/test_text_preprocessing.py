"""Test module for text preprocessors."""
from joblib.testing import parametrize
import pytest

from data.file_loaders import TextPage
from data.text_preprocessing import Languages, SpacyNLPTextPreprocessor
from test.integration.data.conf import example_sentence_1, example_sentence_2, example_sentence_3, example_sentence_4


@pytest.fixture
def polish_text_preprocessor():
    return SpacyNLPTextPreprocessor(Languages.polish)


test_doc = [TextPage(0, example_sentence_1 + example_sentence_2), TextPage(1, example_sentence_3 + example_sentence_4)]


def test_split_test(polish_text_preprocessor):
    assert polish_text_preprocessor.count_sentences_in_text(example_sentence_1 + example_sentence_2) == 1


def test_split_pages_on_sentences(polish_text_preprocessor):
    sentences = polish_text_preprocessor.split_pages_on_sentences(test_doc)
    assert len(sentences) == 3


@parametrize("chunk_size, number_of_expected_chunks", [(10, 2), (11, 2), (3, 7)])
def test_get_chunks_of_sentences(polish_text_preprocessor, chunk_size, number_of_expected_chunks):
    sentences = [example_sentence_1] * 20
    assert len(list(polish_text_preprocessor.get_sentences_chunks(sentences, chunk_size))) == number_of_expected_chunks
