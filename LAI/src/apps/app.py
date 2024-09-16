"""Implementation of RAG pipeline."""
from langcodes import Language

from apps.base_app import BaseApp
from data.file_loaders import BaseDataProcessor, PDFDataProcessor, TextPage
from data.text_preprocessing import BaseTextProcessor, SpacyNLPTextPreprocessor, Languages
from database.chroma_db import TempChromaDB, SentenceEmbeddingFunction
from models.base_embedding_model import EmbeddingModel
from models.embedding_model.sentence_transformer_model import SentenceTransformerModel


class AlphaLAI(BaseApp):
    """Implementation of alfa version of LAI app."""

    def __init__(self, language: Languages.polish):
        """Init the app."""
        self.language = language
        self.db = self._get_database()

    @staticmethod
    def _get_embedding_model_name() -> str:
        return "all-mpnet-base-v2"

    def _get_embedding_model(self) -> EmbeddingModel:
        return SentenceTransformerModel(self._get_embedding_model_name())

    def _get_embedding_function(self) -> SentenceEmbeddingFunction:
        return SentenceEmbeddingFunction(self._get_embedding_model())

    def _get_database(self) -> TempChromaDB:
        return TempChromaDB(collection_name='alpha_lai_db', embedding_function=self._get_embedding_function())

    def _get_input_data_loader(self) -> BaseDataProcessor:
        return PDFDataProcessor()

    def _get_text_preprocessor(self) -> SpacyNLPTextPreprocessor:
        return SpacyNLPTextPreprocessor(self.language)

    def load_file(self, file_path: str) -> list[TextPage]:
        file_loader = self._get_input_data_loader()
        return file_loader.load_data(file_path)

    def insert_data_to_db(self, data: list[TextPage]) -> None:
        text_preprocessor = self._get_text_preprocessor()
        id = 0
        for text_page in data:
            text_chunks = text_preprocessor.split_text_on_sentences(text_page.text)
            for sentece_chunk in text_preprocessor.get_sentences_chunks(text_chunks, 3):
                self.db.add_to_db(documents=sentece_chunk,
                                  metadatas={'page_nr': str(text_page.page_number)},
                                  ids=f"id_{id}")
                id += 1



