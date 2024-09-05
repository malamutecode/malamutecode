"""Module for processing input txt data."""
from abc import ABC, abstractmethod

import fitz
from tqdm import tqdm


class BaseDataProcessor(ABC):
    """Base class for data processing."""

    @classmethod
    @abstractmethod
    def load_data(cls, file_path: str):
        pass


class PDFDataProcessor(BaseDataProcessor):
    """Class to parse PDF data."""


    @staticmethod
    def text_formatter(text: str) -> str:
        """Performs minor formatting on text."""
        cleaned_text = text.replace("\n", " ").strip()
        return cleaned_text

    @classmethod
    def load_data(cls, pdf_path: str) -> list[dict]:
        """Open and read PDF."""
        doc = fitz.open(pdf_path)
        pages_and_texts = []
        for page_number, page in tqdm(enumerate(doc)):
            text = page.get_text()
            text = cls.text_formatter(text=text)
            pages_and_texts.append({"page_number": page_number - 41,
                                    "page_char_count": len(text),
                                    "page_word_count": len(text.split(" ")),
                                    "page_setence_count_raw": len(text.split(". ")),
                                    "page_token_count": len(text) / 4,  # 1 token = ~4 characters
                                    "text": text})
        return pages_and_texts
