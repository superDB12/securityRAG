import os
from dotenv import load_dotenv
# load_dotenv()
# USER_AGENT = os.environ.get("USER_AGENT")
import bs4
from langchain import hub
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from database_access.session_factory import SessionFactory
from database_access.splitCrud import SplitCRUD
from retriever.retriever import DocumentSearcher

# This code is working but I don't understand it very well.


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

class Generator:
    def __init__(self):
        self.split_crud = SplitCRUD(SessionFactory())
        self.doc_searcher = DocumentSearcher()
        self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")

    def retrieve(self, state: State):
        retrieved_docs = self.doc_searcher.search_similar_splits(state["question"])
        return {"context": retrieved_docs}

    def generate(self, state:State):
        docs_content = "\n\n".join(self.split_crud.get_split_content(doc.SplitID) for doc in state[
            "context"])
        prompt = f"Question: {state['question']}\n\nRelevant Information:\n{docs_content}\n\nAnswer the question based on the relevant information provided."
        generated_response = self.llm.invoke(prompt)
        return {"answer": generated_response.content}

    def build_graph(self):
        graph_builder = StateGraph(State).add_sequence([self.retrieve, self.generate])
        graph_builder.add_edge(START, "retrieve")
        return graph_builder.compile()

if __name__ == "__main__":
    generator = Generator()
    response = generator.build_graph().invoke({"question": "What are the most recent topics "
                                                           "discussed by Steve in episode 1013?"})
    print(f"Response: {response["answer"]}")

