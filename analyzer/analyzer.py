import os
# import re # No longer used
# from datetime import datetime # No longer used
import logging
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from database_access.docCrud import DocumentCRUD
from database_access.session_factory import SessionFactory
from database_access.splitCrud import SplitCRUD
from utils.secret_manager import get_secret

load_dotenv()
assert os.environ.get("SPLIT_LENGTH") is not None, "You need an SPLIT_LENGTH"
assert os.environ.get("SPLIT_OVERLAP") is not None, "You need an SPLIT_OVERLAP"

class DocumentAnalyzer:
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

        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        sessionFactory = SessionFactory()
        self.doc_crud = DocumentCRUD(sessionFactory)
        self.split_crud = SplitCRUD(sessionFactory)

    # Removed extract_date, extract_series, extract_episode, extract_title, extract_hosts
    # Removed insert_episode_data

    def load_splits_and_vectors (self):
        documents = self.doc_crud.get_all_documents()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=int(os.environ.get("SPLIT_LENGTH")), chunk_overlap=int(os.environ.get("SPLIT_OVERLAP")))

        for doc in documents:
            if doc.DocContent is not None and doc.EpisodeAirDate is not None and not doc.Processed:
                logging.info(f'Printing length of doc: {len(doc.DocContent)}')
                split_start_offset = 0
                episode_number = doc.EpisodeNumber
                episode_title = doc.EpisodeTitle
                episode_date = doc.EpisodeAirDate
                # episode_metadata= (f"Episode Number: {episode_number}, Episode Title: "
                #                    f"{episode_title}, Episode Date: {episode_date} : ")
                episode_metadata = ''
                splits = text_splitter.split_text(doc.DocContent)
                for split in splits:
                    vector = self.embeddings.embed_query(episode_metadata + split)
                    # David and John found that embedding the metadata may throw off the symantec meaning.
                    # We also confirmed that we don't need to normalize the vectors.
                    # vector = self.embeddings.embed_query(episode_metadata + split)
                    # normalized_vector = self.split_crud.normalize_split_vectors(vector)
                    #TODO: ensure that the metadata is actually being appended to the split content
                    logging.info(f"Episode metadata: {episode_metadata}")
                    logging.info(f"Split content: {split}")
                    logging.info(f"Generated vector of length {len(vector)} for split")

                    self.split_crud.add_split_document(doc.DocID, split_start_offset, len(split),
                                                       vector, SplitContent=split)
                    logging.info(f"Added split document for DocID {doc.DocID}")
                    logging.info(f"Split start offset: {split_start_offset}, split length: {len(split)}")
                    split_start_offset = split_start_offset + len(split)
                self.doc_crud.update_document(doc.DocID, processed=True)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # logger = logging.getLogger() # This line can be removed if logger is not explicitly used. logging.info will use the root logger.
    logging.info("Starting document analyzer...")
    analyzer = DocumentAnalyzer()
    # analyzer.insert_episode_data() # Removed this line
    analyzer.load_splits_and_vectors()
    logging.info("Document analyzer finished")