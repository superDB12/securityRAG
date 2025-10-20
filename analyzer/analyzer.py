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
from sentence_transformers import SentenceTransformer
from database_access.embeddingsCrud import EmbeddingsCRUD

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
        self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings_crud = EmbeddingsCRUD(sessionFactory)

    # Removed extract_date, extract_series, extract_episode, extract_title, extract_hosts
    # Removed insert_episode_data

    def load_splits_and_vectors (self):
        documents = self.doc_crud.get_all_documents()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=int(os.environ.get("SPLIT_LENGTH")), chunk_overlap=int(os.environ.get("SPLIT_OVERLAP")))

        for doc in documents:
            if doc.DocContent is not None and doc.EpisodeAirDate is not None and not doc.Processed:
                # logging.info(f'Printing length of doc: {len(doc.DocContent)}')
                split_start_offset = 0
                episode_number = doc.EpisodeNumber
                episode_title = doc.EpisodeTitle
                episode_date = doc.EpisodeAirDate
                episode_metadata= (f"Episode Number: {episode_number}, Episode Title: "
                                   f"{episode_title}, Episode Date: {episode_date} : ")
                # episode_metadata = ''
                splits = text_splitter.split_text(doc.DocContent)
                for split in splits:
                    logging.info(f"Episode metadata: {episode_metadata}")
                    logging.info(f"Split content: {split}")

                    # do NOT assume splits are continuous.  Find the start offset of the split in the original document
                    actual_split_start_offset = doc.DocContent.find(split)
                    if actual_split_start_offset == -1:
                        assert False, "Could not find split in original document content"
                    else:
                        split_start_offset = actual_split_start_offset

                    if not self.split_crud.does_split_exist(doc.DocID, split_start_offset):
                        # add the new split to the database
                        new_split_id = self.split_crud.add_split_document(
                            doc.DocID,
                            split_start_offset,
                            len(split),
                            SplitContent=split
                        )

                        # Verify that the split content stored matches the original split - prevents nasty bug that's hard to find later
                        split_content_from_offset = self.split_crud.get_split_content(new_split_id)
                        split_content = split
                        if split_content != split_content_from_offset:
                            assert False, f"Mismatch in split content for SplitID {new_split_id}

                        try:
                            # Store OpenAI embedding alongside SBERT in the embeddings table
                            if new_split_id is not None:
                                self.embeddings_crud.add_embedding(
                                    split_id=new_split_id,
                                    doc_id=doc.DocID,
                                    split_text=split
                                )
                            else:
                                logging.warning("add_split_document did not return a SplitID; skipping EmbeddingsCRUD storage for this split.")
                        except Exception as e:
                            logging.error(f"Failed to store embeddings in embeddings table: {e}")
                    else:  # Split already exists
                        logging.warning(f"Split already exists for DocID {doc.DocID} at offset {split_start_offset}, update the split and the embeddings")

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