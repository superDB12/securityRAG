# This script populates the SBERT embeddings for document splits that currently lack them.
# created during development of SBERT embeddings likely will not be needed again.


import os
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

logging.info("initializing DB Session and Factories")
sessionFactory = SessionFactory()
doc_crud = DocumentCRUD(sessionFactory)
split_crud = SplitCRUD(sessionFactory)

logging.info("initializing OpenAI Embeddings model")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

logging.info("initializing SBERT Embeddings model")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings_crud = EmbeddingsCRUD(sessionFactory)

logging.info("Fetching all embeddings from embeddings table")
all_embeddings = embeddings_crud.get_all_embeddings()
# logging.info(f"Total embeddings in embeddings table: {len(all_embeddings)}")

for embedding in all_embeddings:
    logging.info(f"Embedding ID: {embedding.EmbeddingID}")
    if embedding.SBERTEmbedding is not None:
        print(f"  SBERT Embedding already exists for EmbeddingID: {embedding.EmbeddingID}, skipping.")
    else:
        split = split_crud.get_one_split(embedding.SplitID)
        e=embedding.SplitID
        s=split.SplitID
        if e != s:
            logging.error(f"Mismatch between Embedding SplitID {e} and Split SplitID {s}!")
        split_text = split_crud.get_split_content(embedding.SplitID)
        # logging.info(f"SplitID: {split.SplitID} DocID: {split.DocID} Split Content: {spit_text}")
        logging.info(f"SplitID: {split.SplitID} DocID: {split.DocID} Calculated SBERT Embedding:")

        embeddings_crud.update_only_SBERT_embedding(embedding_id=embedding.EmbeddingID, split_text=split_text)
    logging.info('-------------------------------------------------')