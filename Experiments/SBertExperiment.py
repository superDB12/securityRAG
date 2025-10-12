from sentence_transformers import SentenceTransformer
from database_access.embeddingsCrud import EmbeddingsCRUD
import logging

class DocumentAnalyzer:
    def __init__(self, sessionFactory):
        self.split_crud = SplitCRUD(sessionFactory)
        self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings_crud = EmbeddingsCRUD(sessionFactory)
        # other init code...

    def load_splits_and_vectors(self, doc):
        # code before...
        for split_start_offset, split in some_split_generator(doc):
            vector = self.embeddings.embed_query(split)
            sbert_vector = self.sbert_model.encode(split).tolist()

            if not self.split_crud.split_exists(doc.DocID, split_start_offset):
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
                            embedding=sbert_vector,
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
                                embedding=sbert_vector,
                                embedding_model="sBert"
                            )
                        else:
                            logging.warning("Could not resolve existing SplitID; skipping EmbeddingsCRUD storage for this split.")
                    else:
                        logging.warning("SplitCRUD.get_split_id_by_doc_and_offset not available; cannot store embeddings for existing split.")
                except Exception as e:
                    logging.error(f"Failed to store embeddings for existing split: {e}")
        # code after...