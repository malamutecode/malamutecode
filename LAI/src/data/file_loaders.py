"""Module for processing input txt data."""
from abc import ABC, abstractmethod

import fitz
from attr import dataclass
from tqdm import tqdm


class BaseDataProcessor(ABC):
    """Base class for data processing."""

    @classmethod
    @abstractmethod
    def load_data(cls, file_path: str):
        pass


@dataclass
class TextPage:
    """Class for keeping page's text and relevant data."""

    page_number: int
    text: str
    page_char_count: int = 0
    page_word_count: int = 0
    page_setence_count_raw: int = 0
    page_token_count: int = 0

    def __post_init__(self):
        self.page_char_count = len(self.text.split(" "))
        self.page_word_count = len(self.text.split(". "))
        self.page_setence_count_raw = len(self.text.split(". "))
        self.page_token_count = int(len(self.text) / 4)


class PDFDataProcessor(BaseDataProcessor):
    """Class to parse PDF data."""


    @staticmethod
    def text_formatter(text: str) -> str:
        """Performs minor formatting on text."""
        cleaned_text = text.replace("\n", " ").strip()
        return cleaned_text

    @classmethod
    def load_data(cls, pdf_path: str) -> list[TextPage]:
        """Open and read PDF."""
        doc = fitz.open(pdf_path)
        pages_and_texts = []
        for page_number, page in tqdm(enumerate(doc)):
            text = page.get_text()
            text = cls.text_formatter(text=text)
            pages_and_texts.append(TextPage(page_number=page_number, text=text))
        return pages_and_texts
