import unittest
import logging
import os

from utils.secret_manager import get_secret

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer

from database_access.session_factory import SessionFactory
from database_access.splitCrud import SplitCRUD
from database_access.embeddingsCrud import EmbeddingsCRUD, Embeddings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EnsureEmbeddings(unittest.TestCase):
    def setUp(self):
        load_dotenv()
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

        self.session_factory = SessionFactory()
        self.split_crud = SplitCRUD(self.session_factory)
        self.embeddings_crud = EmbeddingsCRUD(self.session_factory)
        self.openai_embedder = OpenAIEmbeddings(model="text-embedding-3-large")
        self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

    def test_ensure_all_splits_have_embeddings(self):
        """
        Iterates through all splits and ensures that both OpenAI and SBERT embeddings exist.
        If an embedding is missing, it is generated and saved.
        """
        # Use a single session for the whole operation to avoid DetachedInstanceError
        with self.session_factory.get_session() as session:
            all_splits = self.split_crud.get_all_splits()
            logging.info(f"Found {len(all_splits)} splits to check.")

            for split in all_splits:
                # Check for existing embeddings
                existing_embeddings = session.query(Embeddings).filter(Embeddings.SplitID == split.SplitID).all()
                found_models = {emb.EmbeddingModel for emb in existing_embeddings}

                # Check for OpenAI embedding
                if "OpenAI" not in found_models:
                    logging.info(f"Missing OpenAI embedding for SplitID {split.SplitID}. Generating...")
                    openai_vector = self.openai_embedder.embed_query(split.SplitContent)
                    self.embeddings_crud.add_embedding(
                        split_id=split.SplitID,
                        doc_id=split.DocID,
                        embedding=openai_vector,
                        embedding_model="OpenAI"
                    )
                    logging.info(f"OpenAI embedding added for SplitID {split.SplitID}.")

                # Check for SBERT embedding
                if "sBert" not in found_models:
                    logging.info(f"Missing SBERT embedding for SplitID {split.SplitID}. Generating...")
                    sbert_vector = self.sbert_model.encode(split.SplitContent).tolist()
                    sbert_vector_padded = sbert_vector + [0.0] * (3072 - len(sbert_vector))
                    self.embeddings_crud.add_embedding(
                        split_id=split.SplitID,
                        doc_id=split.DocID,
                        embedding=sbert_vector_padded,
                        embedding_model="sBert"
                    )
                    logging.info(f"SBERT embedding added for SplitID {split.SplitID}.")

                if "OpenAI" in found_models and "sBert" in found_models:
                    logging.info(f"Both OpenAI and SBERT embeddings already exist for SplitID {split.SplitID}.")

if __name__ == '__main__':
    unittest.main()