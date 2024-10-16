"""Test fpr file loaders module."""
import os

from src.data.file_loaders import PDFDataProcessor
from src.data.utils.file_downloaders import FileDownloader

from test import CACHE_DIR

EXAMPLE_SATEMENT_FILENAME_1 = 'example_statement_1.pdf'


def pdf_parsing_test():
    cached_file_path = os.path.join(CACHE_DIR, EXAMPLE_SATEMENT_FILENAME_1)
    if not os.path.exists(cached_file_path):
        os.makedirs(CACHE_DIR, exist_ok=True)
        pdf_url = r"""https://orzeczenia.ms.gov.pl/content.pdffile/$002fneurocourt$002fpublished$002f15$002f451005$002
        f0001006$002fW$002f2019$002f000306$002f154510050001006_II_W_000306_2019_Uz_2019-10-31_001-publ.xml?t:ac=pijany
        $0020rowerzysta/154510050001006_II_W_000306_2019_Uz_2019-10-31_001"""
        FileDownloader.download_pdf_from_orzeczenia_ms(pdf_url, cached_file_path)

    loaded_pdf = PDFDataProcessor.load_data(cached_file_path)
    assert len(loaded_pdf) == 5


if __name__ == '__main__':
    pdf_parsing_test()
