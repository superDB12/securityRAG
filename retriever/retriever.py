#Search for similar splits here
#Send query and similar splits to LLM?
#Generate answer and return
#Do we store the answer?
#If we do store the answer, do we store it in the same table as the original document?


import logging
from langchain_openai import OpenAIEmbeddings
from database_access.session_factory import SessionFactory
from database_access.splitCrud import SplitCRUD


class DocumentSearcher:
    def __init__(self):
        self.split_crud = SplitCRUD(SessionFactory())

    def search_similar_splits(self, query_text):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        query_vector = embeddings.embed_query(query_text)
        similar_splits = self.split_crud.get_similar_splits_from_vector(query_vector)
        for split in similar_splits:
            logging.info(f"DocID: {split.DocID} SplitID: {split.SplitID}" )
            logging.info(f"Split content: {self.split_crud.get_split_content(split.SplitID)}")
        # Should we return the query_text or vector here also?
        return similar_splits

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logging.info("Starting document searcher...")
    searcher = DocumentSearcher()
    search = searcher.search_similar_splits("What are some general security practices that Steve "
                                            "recommends?")
    logging.info("Document searcher finished")