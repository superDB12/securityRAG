import logging
import os
from langchain.chat_models import init_chat_model
from utils.secret_manager import get_secret
from database_access.requestAndResponseLogCRUD import RequestAndResponseLogCRUD
from database_access.session_factory import SessionFactory
from datetime import datetime

class ConceptExtractor:
    def __init__(self):
        try:
            project_id = os.getenv("PROJECT_ID")
            if not project_id:
                logging.error("PROJECT_ID environment variable not set. Cannot fetch OpenAI API Key.")
                # Depending on desired behavior, could raise an error or let init_chat_model handle it.
            else:
                openai_api_key = get_secret(project_id, "OPENAI_API_KEY")
                os.environ["OPENAI_API_KEY"] = openai_api_key
                logging.info("Successfully fetched and set OPENAI_API_KEY for ConceptExtractor.")
        except Exception as e:
            logging.error(f"Error fetching OPENAI_API_KEY from Secret Manager for ConceptExtractor: {e}")
            # Proceeding, Langchain will raise an error if the key isn't found/set.

        self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        # Initialize RequestAndResponseLogCRUD, assuming SessionFactory is available
        # and the import path is correct.
        # from database_access.session_factory import SessionFactory
        # from database_access.requestAndResponseLogCRUD import RequestAndResponseLogCRUD
        self.request_response_logger = RequestAndResponseLogCRUD(SessionFactory())

    def extract_concept(self, user_question: str) -> str:
        """
        Extracts the core concept from a user's question using multi-shot prompting.
        """
        prompt_template_str = """\
Your task is to extract the core concept from the user's question.
Follow these rules:
- Remove conversational or irrelevant context (e.g., "Can you tell me", "I’m wondering", etc.)
- Output a short noun phrase or query-style phrase.
- Do NOT answer the question — just extract the concept.
- Be concise and accurate.

Here are some examples:

User Question: "Can you tell me what a buffer overflow attack is in cybersecurity?"
Core Concept: buffer overflow attack in cybersecurity

User Question: "What are the differences between HTTP and HTTPS?"
Core Concept: differences between HTTP and HTTPS

User Question: "I’m trying to understand how RAM differs from ROM."
Core Concept: difference between RAM and ROM

User Question: "How does the Java garbage collector work?"
Core Concept: java garbage collection

Now, extract the core concept from the following question:

User Question: "{user_question}"
Core Concept:"""

        # Using RunnableSequence as shown in the Generator class
        # (though PromptTemplate and LLMChain are also viable options and sometimes simpler for direct calls)
        # For direct invocation with a formatted string, we might not even need PromptTemplate here,
        # but using it for consistency with potential future Langchain features.

        # Simplified approach for direct string formatting with f-string,
        # as PromptTemplate here is a bit overkill if we are not using its other features.
        prompt = prompt_template_str.format(user_question=user_question)

        # Invoke the LLM. Assuming the LLM object can be called directly or has an invoke method.
        # Based on generator.py, it seems like `llm.invoke()` is the way.
        # The response from llm.invoke is typically a message object (e.g., AIMessage)
        # and we need to access its content.
        try:
            response_message = self.llm.invoke(prompt)
            # Assuming response_message has a 'content' attribute
            extracted_concept = response_message.content.strip()

            # Log the interaction
            try:
                self.request_response_logger.add_request_and_response_log(
                    request_text=user_question,
                    response_text=extracted_concept, # Storing concept as "response"
                    # request_type="concept_extraction", # If this column exists or is added.
                                                        # Assuming it does not for now.
                    date=datetime.now()
                )
                logging.info(f"Concept extraction interaction logged for question: {user_question[:50]}...")
            except Exception as log_e:
                logging.error(f"Failed to log concept extraction interaction: {log_e}")

            return extracted_concept
        except Exception as e:
            logging.error(f"Error during concept extraction LLM call: {e}")
            # Fallback or re-raise error, depending on desired error handling
            return "Error extracting concept" # Or raise e
