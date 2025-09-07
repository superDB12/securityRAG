# Run thru split_table and transfer openAI embeddings to embeddings table

import logging
import os
from dotenv import load_dotenv
from database_access.session_factory import SessionFactory
from database_access.splitCrud import SplitCRUD
from database_access.embeddingsCrud import EmbeddingsCRUD
from utils.secret_manager import get_secret


def migrateEmbeddings():
    sessionFactory = SessionFactory()
    split_crud = SplitCRUD(sessionFactory)
    embeddings_crud = EmbeddingsCRUD(sessionFactory)

    splits = split_crud.get_all_splits()
    for split in splits:
        if split.SplitVector is not None:
            embeddings_crud.add_embedding(split.SplitID, split.DocID, split.SplitVector,
                                          "openai-text-embedding-3-large")
            logging.info(f"Migrated embedding for SplitID: {split.SplitID}")
        else:
            logging.info(f"No embedding found for SplitID: {split.SplitID}, skipping migration.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logging.info("Starting embedding migration...")
    migrateEmbeddings()
    logging.info("Embedding migration finished")