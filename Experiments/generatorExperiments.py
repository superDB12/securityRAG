# This code works but it is deprecated and seems like it should be done differently

from langchain_core.runnables import Runnable, RunnableSequence
from retriever.retriever import DocumentSearcher
from database_access.docCrud import DocumentCRUD, DatabaseConnection
from database_access.engine_factory import EngineFactory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import logging
from langchain.chat_models import init_chat_model


class Generator:
    def __init__(self):
        self.engine = EngineFactory().get_engine()
        self.doc_crud = DocumentCRUD(DatabaseConnection(self.engine))
        self.doc_searcher = DocumentSearcher()
        self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")

    def generate_response(self, query_text):
        # Search for similar splits
        similar_splits = self.doc_searcher.search_similar_splits(query_text)

        # Prepare the prompt with the query and similar splits
        splits_text = "\n".join([split.SplitContent for split in similar_splits])
        prompt = f"Query: {query_text}\n\nRelevant Information:\n{splits_text}\n\nAnswer the query based on the relevant information provided."

        # Create a LangChain prompt template
        prompt_template = PromptTemplate(template=prompt,
                                         input_variables=["query_text", "splits_text"])

        # Create a LangChain LLMChain
        # LLMChain is deprecated, unsure how to properly use the new API
        # llm_chain = LLMChain(llm=self.llm, prompt=prompt_template)
        sequence = RunnableSequence(prompt_template | self.llm)

        # Get the response from the LLM
        # .run is deprecated, directed to use .invoke instead but .invoke just brings back the
        # text, it is not interpreted in any way
        # response = llm_chain.run({"query_text": query_text, "splits_text": splits_text})
        response = sequence.invoke({"query_text": query_text, "splits_text": splits_text})
        # response = runnable.invoke(
        #     prompt_template.format(query_text=query_text, splits_text=splits_text))
        return response


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logging.info("Starting generator...")
    generator = Generator()
    response = generator.generate_response("What does Steve say about trojan horse attacks?")
    logging.info(f"Response: {response}")
    logging.info("Generator finished")
