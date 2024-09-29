"""Implementation of RAG pipeline."""
import huggingface_hub
import torch
from transformers import BitsAndBytesConfig

from apps.base_app import BaseApp
from data.file_loaders import BaseDataProcessor, PDFDataProcessor, TextPage
from data.text_preprocessing import Languages, SpacyNLPTextPreprocessor
from database.chroma_db import SentenceEmbeddingFunction, TempChromaDB
from models.base_embedding_model import EmbeddingModel
from models.embedding_model.sentence_transformer_model import SentenceTransformerModel
from models.llm.hugging_face_llm import HuggingFaceLLM
from models.tokenizer.auto_tokenizer import AutoTokenizerModel


class AlphaLAI(BaseApp):
    """Implementation of alfa version of LAI app."""

    def __init__(self, language: Languages = Languages.polish):
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

    def _get_tokenizer(self) -> AutoTokenizerModel:
        return AutoTokenizerModel("google/gemma-2b-it")

    def _get_llm_model(self) -> HuggingFaceLLM:
        huggingface_hub.login(token='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')
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        return HuggingFaceLLM("google/gemma-2b-it", torch.float16, bnb_config, tokenizer=self._get_tokenizer())

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

        base_prompt = """Based on the following context items, please answer the query.
        Give yourself room to think by extracting relevant passages from the context before answering the query.
        Don't return the thinking, only return the answer.
        Make sure your answers are as explanatory as possible.
        Use the following examples as reference for the ideal answer style.
        Example 1:
        Query: What are the fat-soluble vitamins?
        Answer: The fat-soluble vitamins include Vitamin A, Vitamin D, Vitamin E, and Vitamin K.
        These vitamins are absorbed along with fats in the diet and can be stored in the body's fatty tissue and liver
        for later use. Vitamin A is important for vision, immune function, and skin health. Vitamin D plays a critical
        role in calcium absorption and bone health. Vitamin E acts as an antioxidant, protecting cells from damage.
        Vitamin K is essential for blood clotting and bone metabolism.
        Example 2:
        Query: What are the causes of type 2 diabetes?
        Answer: Type 2 diabetes is often associated with overnutrition, particularly the overconsumption of calories
        leading to obesity. Factors include a diet high in refined sugars and saturated fats, which can lead to insulin
        resistance, a condition where the body's cells do not respond effectively to insulin. Over time, the pancreas
        cannot produce enough insulin to manage blood sugar levels, resulting in type 2 diabetes. Additionally,
        excessive caloric intake without sufficient physical activity exacerbates the risk by promoting weight
        gain and fat accumulation, particularly around the abdomen, further contributing to insulin resistance.
        Example 3:
        Query: What is the importance of hydration for physical performance?
        Answer: Hydration is crucial for physical performance because water plays key roles in maintaining blood volume,
        regulating body temperature, and ensuring the transport of nutrients and oxygen to cells. Adequate hydration is
        essential for optimal muscle function, endurance, and recovery. Dehydration can lead to decreased performance,
        fatigue, and increased risk of heat-related illnesses, such as heat stroke. Drinking sufficient water before,
        during, and after exercise helps ensure peak physical performance and recovery.
        Now use the following context items to answer the user query:
        {context}
        Relevant passages: <extract relevant passages from the context here>
        User query: {query}
        Answer:"""
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
        model = self._get_llm_model()
        llm_outputs = model.generate_from_packed_prompt(batch_encoded_prompt)
        output_text = tokenizer.decode(llm_outputs[0])
        return output_text
