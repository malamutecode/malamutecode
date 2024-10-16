"""Module for text preprocessors."""
from abc import ABC, abstractmethod
from collections.abc import Generator

from attr import dataclass
import spacy
from tqdm import tqdm

from data.file_loaders import TextPage
from logger import log


@dataclass
class Languages:
    """Languages supported by the text processor."""

    polish = 'polish'
    english = 'english'


class BaseTextProcessor(ABC):
    """Base class for data processing."""

    @abstractmethod
    def split_text_on_sentences(self, text: str) -> list[str]:
        pass

    @abstractmethod
    def split_pages_on_sentences(self, pages: list[TextPage]) -> list[str]:
        pass

    @staticmethod
    def get_sentences_chunks(sentences: list[str], chunk_size: int) -> Generator:
        """
        Joint sentences into text.
        :param: chunk_size: Number of sentences to join into one paragraph.
        """
        for sentence_number in range(0, len(sentences), chunk_size):
            text_paragraph = " ".join(sentences[sentence_number: sentence_number + chunk_size])
            text_paragraph = text_paragraph.replace("  ", " ").strip()
            yield text_paragraph


class SpacyNLPTextPreprocessor(BaseTextProcessor):
    """Class for text preprocessing with spacy lib"""

    def __init__(self, language: Languages):
        """Initialize the text preprocessor."""
        self.nlp = self._get_nlp_for_language(language)
        self._init_sentencizer()

    @staticmethod
    def _get_nlp_for_language(language: Languages) -> spacy.language.Language:
        match language:
            case Languages.polish:
                try:

                    return spacy.load('pl_core_news_lg')
                except OSError:
                    log.error("Missing model - install with 'python -m spacy download pl_core_news_lg'")
                    raise
            case Languages.english:
                from spacy.lang.en import English
                return English()
            case _:
                raise NotImplementedError(f"Given language: {language} is not supported.")

    def _init_sentencizer(self):
        """Init NLP pipeline with sentencizer to divide text into sentences."""
        self.nlp.add_pipe('sentencizer')

    def split_text_on_sentences(self, text: str) -> list[str]:
        return [str(sentence) for sentence in self.nlp(text).sents]

    def split_pages_on_sentences(self, pages: list[TextPage]) -> list[str]:
        sentences = []
        for page in tqdm(pages, desc="Dividing text pages to sentences"):
            sentences.extend(self.split_text_on_sentences(page.text))
        return sentences

    def count_sentences_in_text(self, text: str) -> int:
        return len(list(self.nlp(text).sents))
