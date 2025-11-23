#Search for similar splits here
#Send query and similar splits to LLM?
#Generate answer and return
#Do we store the answer?
#If we do store the answer, do we store it in the same table as the original document?


import logging
from langchain_openai import OpenAIEmbeddings
from database_access.session_factory import SessionFactory
from database_access.splitCrud import SplitCRUD
from sentence_transformers import SentenceTransformer
from database_access import embeddingsCrud

class DocumentSearcher:
    def __init__(self):
        self.split_crud = SplitCRUD(SessionFactory())

    def search_similar_splits_using_OpenAI(self, query_text) -> list:
        # embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        # query_vector = embeddings.embed_query(query_text)
        query_vector = embeddingsCrud.calculate_openAI_embedding(query_text)
        similar_splits = self.split_crud.get_similar_splits_from_embeddings(query_vector, embedding_model='OpenAI')
        logging.info(f"Found {len(similar_splits)} similar splits for query: {query_text}")
        for split in similar_splits:
            logging.info(f"DocID: {split.DocID} SplitID: {split.SplitID}" )
            logging.info(f"Split content: {self.split_crud.get_split_content(split.SplitID)}")
        # Should we return the query_text or vector here also?
        return similar_splits

    def search_similar_splits_using_SBERT(self, query_text) -> list:
        # model = SentenceTransformer("all-MiniLM-L6-v2")
        # query_vector = model.encode(query_text).tolist()
        # # pad to 3072 dims to match stored SBERT vectors
        # if len(query_vector) < 3072:
        #     query_vector = query_vector + [0.0] * (3072 - len(query_vector))
        query_vector = embeddingsCrud.calculate_SBERT_embedding(query_text)
        similar_splits = self.split_crud.get_similar_splits_from_embeddings(query_vector, embedding_model='sBert')
        logging.info(f"Found {len(similar_splits)} similar splits for query: {query_text}")
        for split in similar_splits:
            logging.info(f"DocID: {split.DocID} SplitID: {split.SplitID}")
            logging.info(f"Split content: {self.split_crud.get_split_content(split.SplitID)}")
        return similar_splits

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logging.info("Starting document searcher...")
    searcher = DocumentSearcher()
    search1 = searcher.search_similar_splits_using_OpenAI("give me key topics for 5 page technical report about Salt Typhoon.")
    logging.info("Finished search using OpenAI")
    logging.info("Starting search using SBERT")
    search2 = searcher.search_similar_splits_using_SBERT("give me key topics for 5 page technical report about Salt Typhoon.")
    logging.info("Finished search using SBERT")

    logging.info("Comparing results - OpenAI returned {} results, SBERT returned {} results".format(len(search1), len(search2)))

    logging.info("Document searcher finished")