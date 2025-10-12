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
                    if not self.split_crud.does_split_exist(doc.DocID, split_start_offset):
                        logging.info(f"Generating vector for split")
                        vector = self.embeddings.embed_query(split)
                        sbert_vector = self.sbert_model.encode(split).tolist()
                        # Pad SBERT (384-dim) to 3072-dim to match the embeddings table schema
                        if isinstance(sbert_vector, list):
                            sbert_len = len(sbert_vector)
                        else:
                            try:
                                sbert_len = len(sbert_vector)
                            except Exception:
                                sbert_len = 0
                        if sbert_len < 3072:
                            sbert_vector_padded = sbert_vector + [0.0] * (3072 - sbert_len)
                        elif sbert_len == 3072:
                            sbert_vector_padded = sbert_vector
                        else:
                            logging.warning(f"SBERT vector length {sbert_len} exceeds table dimension 3072; truncating.")
                            sbert_vector_padded = sbert_vector[:3072]
                        # David and John found that embedding the metadata may throw off the symantec meaning.
                        # We also confirmed that we don't need to normalize the vectors.
                        # vector = self.embeddings.embed_query(episode_metadata + split)
                        # normalized_vector = self.split_crud.normalize_split_vectors(vector)
                        logging.info(f"Generated vector of length {len(vector)} for split")
                        logging.info(f"Adding split document for DocID {doc.DocID}")
                        new_split_id = self.split_crud.add_split_document(
                            doc.DocID,
                            split_start_offset,
                            len(split),
                            vector,
                            SplitContent=split
                        )
                        try:
                            # Store OpenAI embedding alongside SBERT in the embeddings table
                            if new_split_id is not None:
                                self.embeddings_crud.add_embedding(
                                    split_id=new_split_id,
                                    doc_id=doc.DocID,
                                    embedding=vector,
                                    embedding_model="OpenAI"
                                )
                                self.embeddings_crud.add_embedding(
                                    split_id=new_split_id,
                                    doc_id=doc.DocID,
                                    embedding=sbert_vector_padded,
                                    embedding_model="sBert"
                                )
                            else:
                                logging.warning("add_split_document did not return a SplitID; skipping EmbeddingsCRUD storage for this split.")
                        except Exception as e:
                            logging.error(f"Failed to store embeddings in embeddings table: {e}")
                    else:
                        logging.info(f"Split already exists for DocID {doc.DocID} at offset {split_start_offset}, skipping embedding and insertion into DB.")
                        try:
                            # If your SplitCRUD exposes a way to fetch the split id by (DocID, offset), use it here.
                            if hasattr(self.split_crud, "get_split_id_by_doc_and_offset"):
                                existing_split_id = self.split_crud.get_split_id_by_doc_and_offset(doc.DocID, split_start_offset)
                                if existing_split_id is not None:
                                    self.embeddings_crud.add_embedding(
                                        split_id=existing_split_id,
                                        doc_id=doc.DocID,
                                        embedding=vector,
                                        embedding_model="OpenAI"
                                    )
                                    self.embeddings_crud.add_embedding(
                                        split_id=existing_split_id,
                                        doc_id=doc.DocID,
                                        embedding=sbert_vector_padded,
                                        embedding_model="sBert"
                                    )
                                else:
                                    logging.warning("Could not resolve existing SplitID; skipping EmbeddingsCRUD storage for this split.")
                            else:
                                logging.warning("SplitCRUD.get_split_id_by_doc_and_offset not available; cannot store embeddings for existing split.")
                        except Exception as e:
                            logging.error(f"Failed to store embeddings for existing split: {e}")

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