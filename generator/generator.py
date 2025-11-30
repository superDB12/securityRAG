from datetime import datetime

from langchain_core.runnables import Runnable, RunnableSequence
from database_access.splitCrud import SplitCRUD
from database_access.requestAndResponseLogCRUD import RequestAndResponseLogCRUD
from retriever.retriever import DocumentSearcher
from database_access.session_factory import SessionFactory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import logging
from langchain.chat_models import init_chat_model
import os # Added for os.environ and os.getenv
from utils.secret_manager import get_secret # Added for fetching secrets


class Generator:
    def __init__(self):
        # Fetch and set OPENAI_API_KEY from Secret Manager
        try:
            project_id = os.getenv("PROJECT_ID")
            if not project_id:
                logging.error("PROJECT_ID environment variable not set. Cannot fetch OpenAI API Key.")
                # Or raise an error, depending on desired behavior
                # For now, we'll let init_chat_model fail if key is not found through other means
            else:
                openai_api_key = get_secret(project_id, "OPENAI_API_KEY")
                os.environ["OPENAI_API_KEY"] = openai_api_key
                logging.info("Successfully fetched and set OPENAI_API_KEY.")
        except Exception as e:
            # Log the error and proceed. Langchain will raise an error if the key isn't found.
            logging.error(f"Error fetching OPENAI_API_KEY from Secret Manager: {e}")
            # Potentially raise an error here if the API key is critical for startup

        self.split_crud = SplitCRUD(SessionFactory())
        self.doc_searcher = DocumentSearcher()
        self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        self.request_response = RequestAndResponseLogCRUD(SessionFactory())

    def generate_response(self, query_text):
        # Search for similar splits
        similar_splits = self.doc_searcher.search_similar_splits_using_OpenAI(query_text)

        # Prepare the prompt with the query and similar splits
        splits_text = "\n".join([self.split_crud.get_split_content(split.SplitID) for split in
                                                                   similar_splits])
        prompt = (f"Query: {query_text}\n\nRelevant Information:\n{splits_text}\n\nAnswer the "
                  f"query based on the relevant information provided. Say that the information is not available if you are not able to find suitable response")

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
        # self.request_response.add_request_and_response_log(query_text, response.content,
        #                                                    date=datetime.now())
        return response.content

        # return response.content  # Assuming response is a string, if it's an object,

    def generate_response_from_query_and_concept(self, user_query:str, concept:str) -> str:
        # Search for similar splits
        similar_splits = self.doc_searcher.search_similar_splits_using_OpenAI(concept)

        # Prepare the prompt with the query and similar splits
        splits_text = "\n".join([self.split_crud.get_split_content(split.SplitID) for split in
                                                                   similar_splits])
        prompt = (f"Query: {user_query}\n\nRelevant Information:\n{splits_text}\n\nAnswer the "
                  f"query based on the relevant information provided. Say that the information is not available if you are not able to find suitable response")

        # Create a LangChain prompt template
        prompt_template = PromptTemplate(template=prompt,
                                         input_variables=["user_query", "splits_text"])

        # Create a LangChain LLMChain
        # LLMChain is deprecated, unsure how to properly use the new API
        # llm_chain = LLMChain(llm=self.llm, prompt=prompt_template)
        sequence = RunnableSequence(prompt_template | self.llm)

        # Get the response from the LLM
        # .run is deprecated, directed to use .invoke instead but .invoke just brings back the
        # text, it is not interpreted in any way
        # response = llm_chain.run({"query_text": query_text, "splits_text": splits_text})
        response = sequence.invoke({"user_query": user_query, "splits_text": splits_text})
        # response = runnable.invoke(
        #     prompt_template.format(query_text=query_text, splits_text=splits_text))
        # self.request_response.add_request_and_response_log(query_text, response.content,
        #                                                    date=datetime.now())
        return response.content


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logging.info("Starting generator...")
    generator = Generator()
    response = generator.generate_response("What episode does Steve talk about TOAD?")
    logging.info(f"Response: {response}")
    logging.info("Generator finished")
