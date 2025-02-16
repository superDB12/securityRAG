#Search for similar splits here
#Send query and similar splits to LLM?
#Generate answer and return
#Do we store the answer?
#If we do store the answer, do we store it in the same table as the original document?


import logging
from langchain_openai import OpenAIEmbeddings
from database_access.docCrud import DocumentCRUD, DatabaseConnection
from database_access.engine_factory import EngineFactory


class DocumentSearcher:
    def __init__(self):
        self.engine = EngineFactory().get_engine()
        self.doc_crud = DocumentCRUD(DatabaseConnection(self.engine))

    def search_similar_splits(self, query_text):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        query_vector = embeddings.embed_query(query_text)
        similar_splits = self.doc_crud.get_similar_vectors(query_vector)
        for split in similar_splits:
            logging.info(f"SplitID: {split.SplitID}")
        # Should we return the query_text or vector here also?
        return similar_splits

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logging.info("Starting document searcher...")
    searcher = DocumentSearcher()
    search = searcher.search_similar_splits("What does Steve say about trojan horse attacks?")
    logging.info("Document searcher finished")