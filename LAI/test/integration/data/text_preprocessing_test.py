"""Test module for text preprocessors."""
from collections import namedtuple

from data.file_loaders import TextPage
from data.text_preprocessing import Languages, SpacyNLPTextPreprocessor
from test.integration.data.conf import example_sentence_1, example_sentence_2, example_sentence_3, example_sentence_4

test_doc = [TextPage(0, example_sentence_1 + example_sentence_2), TextPage(1, example_sentence_3 + example_sentence_4)]
ChunkSizeNum = namedtuple("ChunkSizeNum", ["chunk_size", "number_of_expected_chunks"])


def get_polish_text_preprocessor():
    return SpacyNLPTextPreprocessor(Languages.polish)


def split_test(polish_text_preprocessor: SpacyNLPTextPreprocessor) -> None:
    assert polish_text_preprocessor.count_sentences_in_text(example_sentence_1 + example_sentence_2) == 1


def split_pages_on_sentences_test(polish_text_preprocessor: SpacyNLPTextPreprocessor) -> None:
    sentences = polish_text_preprocessor.split_pages_on_sentences(test_doc)
    assert len(sentences) == 3


def get_chunks_of_sentences_test(polish_text_preprocessor: SpacyNLPTextPreprocessor, chunk_size: int,
                                 number_of_expected_chunks: int) -> None:
    sentences = [example_sentence_1] * 20
    assert len(list(polish_text_preprocessor.get_sentences_chunks(sentences, chunk_size))) == number_of_expected_chunks


if __name__ == '__main__':
    split_test(get_polish_text_preprocessor())
    split_pages_on_sentences_test(get_polish_text_preprocessor())
    chunk_setups = [ChunkSizeNum(10, 2), ChunkSizeNum(11, 2), ChunkSizeNum(3, 7)]
    for chunk_setup in chunk_setups:
        get_chunks_of_sentences_test(get_polish_text_preprocessor(), chunk_setup.chunk_size,
                                     chunk_setup.number_of_expected_chunks)
