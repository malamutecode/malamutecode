"""Implementation of RAG pipeline."""
from dataclasses import dataclass

import huggingface_hub
import torch
from transformers import BitsAndBytesConfig

from apps.base_app import BaseApp
from data.file_loaders import BaseDataProcessor, PDFDataProcessor, TextPage
from data.text_preprocessing import Languages, SpacyNLPTextPreprocessor
from database.chroma_db import SentenceEmbeddingFunction, TempChromaDB
from models.base_embedding_model import EmbeddingModel
from models.base_tokenizer import Tokenizer
from models.embedding_model.sentence_transformer_model import SentenceTransformerModel
from models.llm.hugging_face_llm import HuggingFaceLLM
from models.propmpts import orzeczenia_prompts
from models.propmpts.prompt_register import get_prompt_registry
from models.tokenizer.auto_tokenizer import AutoTokenizerModel


@dataclass
class AppConfig:
    """Base config for application."""

    hugging_face_token: str = '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'
    language: Languages = Languages.polish
    model_name: str = "speakleash/Bielik-11B-v2.3-Instruct-GPTQ"
    model_bnb_config: BitsAndBytesConfig | None = None
    model_dtype: torch.dtype | None = None
    model_device: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AlphaLAI(BaseApp):
    """Implementation of alfa version of LAI app."""

    def __init__(self, app_config: AppConfig = AppConfig()):
        """Init the app."""
        self.app_config = app_config
        self.db = self._get_database()
        self.model = self._get_llm_model()

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
        return SpacyNLPTextPreprocessor(self.app_config.language)

    def _get_tokenizer(self) -> Tokenizer:
        return self.model.tokenizer

    def _get_llm_model(self) -> HuggingFaceLLM:
        huggingface_hub.login(token=self.app_config.hugging_face_token)
        return HuggingFaceLLM(self.app_config.model_name, model_dtype=self.app_config.model_dtype,
                              quantization_config=self.app_config.model_bnb_config,
                              tokenizer=AutoTokenizerModel(self.app_config.model_name))

    def load_file(self, file_path: str) -> list[TextPage]:
        file_loader = self._get_input_data_loader()
        return file_loader.load_data(file_path)

    def insert_data_to_db(self, data: list[TextPage], sentences_nr: int = 3) -> None:
        """
        Insert loaded data to vector db.

        :param: sentences_nr: Number of sentences to split paragraphs into before putting to database.
        """
        text_preprocessor = self._get_text_preprocessor()
        id = 0
        for text_page in data:
            text_chunks = text_preprocessor.split_text_on_sentences(text_page.text)
            for sentece_chunk in text_preprocessor.get_sentences_chunks(text_chunks, sentences_nr):
                self.db.add_to_db(documents=sentece_chunk,
                                  metadatas={'page_nr': str(text_page.page_number)},
                                  ids=f"id_{id}")
                id += 1

    def query_db(self, query: str, number_of_results: int = 3) -> dict[str, list]:
        results = self.db.collection.query(
            query_texts=[query],
            n_results=number_of_results
        )
        return results

    def prepare_prompt(self, query: str, context_items: list[str]) -> str:
        context = "- " + "\n- ".join([extracted_paragraph for extracted_paragraph in context_items])

        prompts = get_prompt_registry()
        base_prompt = prompts[self.app_config.model_name][Languages.polish]
        base_prompt = base_prompt.format(context=context,
                                         query=query)
        tokenizer = self._get_tokenizer()
        packed_prompt = tokenizer.pack_prompt(base_prompt)
        return packed_prompt


    def prompt_with_rag(self, user_prompt: str) -> str:
        db_data = self.query_db(user_prompt)
        prompt = self.prepare_prompt(query=user_prompt, context_items=db_data['documents'][0])
        tokenizer = self._get_tokenizer()
        batch_encoded_prompt = tokenizer.batch_encode(prompt)
        print('Loading model...')
        print('Generating model response...')
        llm_outputs = self.model.generate_from_packed_prompt(batch_encoded_prompt)
        print('Decoding model response...')
        output_text = tokenizer.decode(llm_outputs[0])
        return output_text








