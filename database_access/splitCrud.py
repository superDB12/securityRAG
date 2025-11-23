from datetime import datetime
import numpy as np
from dns.e164 import query
from dotenv import load_dotenv
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, \
    UniqueConstraint, Boolean, select, func
import os
import logging
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import aliased, Session

from .docCrud import Document, DocumentCRUD
from .session_factory import Base

from database_access.embeddingsCrud import Embeddings

import logging
from logging.config import dictConfig
from typing import Optional



dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'DEBUG',
        'handlers': ['wsgi']
    }
})

load_dotenv()
DEBUG_SPLIT_CONTENT_ENABLED = os.environ.get("DEBUG_SPLIT_CONTENT_ENABLED", "False").lower() == "true"
assert os.environ.get("MAX_SPLITS") is not None, "You have not set MAX_SPLITS."
assert os.environ.get("DIST_THRESHOLD") is not None, "You have not set DIST_THRESHOLD."

#TODO: Add metadata for the split documents

from dataclasses import dataclass
@dataclass
class SplitWithSimilarityDistance:
    SplitID: int
    DocID: int
    SplitContent: str
    SplitStartOffset: int
    SplitLength: int
    SplitCosignDistance: float


#Add a new table for split documents here or in a new file?
class SplitDocument(Base):
    __tablename__ = 'split_documents'
    SplitID = Column(Integer, primary_key=True, autoincrement=True)
    DocID = Column(Integer, ForeignKey('documents.DocID'))
    DateRead = Column(DateTime)
    SplitContent = Column(String)
    SplitStartOffset = Column(Integer)
    SplitLength = Column(Integer)
    #Do we need VectorStored?
    # VectorStored = Column(Boolean, default=False)

    # Conditionally add SplitContent
    if DEBUG_SPLIT_CONTENT_ENABLED:
        SplitContent = Column(String, nullable=True) # Add as nullable String

    __table_args__ = (UniqueConstraint('DocID', 'SplitStartOffset', name='_docid_splitstartoffset_uc'),)

class SplitCRUD:
    def __init__(self, db_connection):
        self.session = db_connection.get_session()
        self.doc_crud = DocumentCRUD(db_connection)
        Base.metadata.create_all(db_connection.get_engine())

#TODO optimize to use offsets and only store the content in the documents table
    def add_split_document(self, doc_id, split_start_offset, split_length,
                           SplitContent=None) -> Optional[int]:
        existing_split = self.session.query(SplitDocument).filter(
            SplitDocument.DocID == doc_id,
            SplitDocument.SplitStartOffset == split_start_offset
        ).first()
        if not existing_split:
            split_doc_args = {
                "DocID": doc_id,
                "DateRead": datetime.now(),
                "SplitStartOffset": split_start_offset,
                "SplitLength": split_length,
            }
            if DEBUG_SPLIT_CONTENT_ENABLED and SplitContent is not None:
                split_doc_args["SplitContent"] = SplitContent

            try:
                new_split_doc: SplitDocument = SplitDocument(**split_doc_args)
                self.session.add(new_split_doc)
                self.session.flush()  # Ensures SplitID is populated
                split_id = new_split_doc.SplitID
                self.session.commit()

                # # add the embeddings
                # from database_access.embeddingsCrud import Embeddings
                #
                # embeddings_crud = Embeddings(self.session)
                # embeddings_crud.add_embedding(
                #     split_id=split_id,
                #     doc_id=doc_id,
                #     embedding=None,
                #     embedding_model="OpenAI"
                # )
                return split_id
            except Exception:
                self.session.rollback()
                logging.exception("Failed to insert split document.")
                raise

        else:
            logging.info(f"Duplicate split found for DocID {doc_id}, skipping insertion.")
            return None

    def update_split_document(self, split_id, doc_id=None, doc_content=None,  split_start_offset=None, split_length=None, vector_stored=None):
        split = self.session.query(SplitDocument).filter(SplitDocument.SplitID == split_id).first()
        if split:
            if doc_id:
                split.DocID = doc_id
            if split_start_offset:
                split.SplitStartOffset = split_start_offset
            if split_length:
                split.SplitLength = split_length
            self.session.commit()

    # TODO: join split and document tables to get the substring of the document content (and remove split content from SplitDocument)
    # TODO: to test the split coming from document, add assert that document content = split content
    def get_one_split(self, split_id):
        return self.session.query(SplitDocument).filter(SplitDocument.SplitID == split_id).first()

    def get_all_splits(self):
        return self.session.query(SplitDocument).all()

    def normalize_split_vectors(self, vector):
        # Normalize the vector to unit length
        norm = np.linalg.norm(vector)
        if norm == 0:
            normalized_vector = vector
        else:
            normalized_vector =  vector / norm
        logging.info(f'\nVector...........: {vector}'
                     f'\nNormalized Vector: {normalized_vector}')
        return normalized_vector
        # return vector

    def get_similar_splits_from_string(self, query_string, top_k=(int(os.environ.get("MAX_SPLITS"))), distance_threshold=float(os.environ.get("DIST_THRESHOLD"))) -> list[SplitWithSimilarityDistance]:
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        query_vector = embeddings.embed_query(query_string)
        logging.info(f'Running get_similar_splits_from_string (via embeddings table) with query: {query_string}')
        result = self.get_similar_splits_from_embeddings(query_vector, embedding_model='OpenAI', top_k=top_k, distance_threshold=distance_threshold)
        logging.info(f'Found {len(result)} similar splits for query: {query_string}')
        return result

    def get_similar_splits_from_vector(self, query_vector, top_k=(int(os.environ.get("MAX_SPLITS"))), distance_threshold=float(os.environ.get("DIST_THRESHOLD"))) -> list[SplitWithSimilarityDistance]:
        logging.info('Delegating get_similar_splits_from_vector to embeddings-table search for model=OpenAI')
        return self.get_similar_splits_from_embeddings(query_vector, embedding_model='OpenAI', top_k=top_k, distance_threshold=distance_threshold)

    def get_similar_splits_from_embeddings(self, query_vector, embedding_model: str, top_k=(int(os.environ.get("MAX_SPLITS"))), distance_threshold=float(os.environ.get("DIST_THRESHOLD"))) -> list[SplitWithSimilarityDistance]:
        """Generic similarity search over the embeddings table for the given EmbeddingModel (e.g., 'OpenAI' or 'sBert')."""
        qsize = len(query_vector)
        logging.info(f"Running embeddings-table similarity search (model={embedding_model}) with {qsize}-dim vector; top_k={top_k}; threshold={distance_threshold}")
        if embedding_model == 'sBert':
            results = (
                self.session.query(
                    SplitDocument,
                    Embeddings.SBERTEmbedding.cosine_distance(query_vector).label('distance')
                )
                .join(Embeddings, Embeddings.SplitID == SplitDocument.SplitID)
                .filter(
                    #Embeddings.EmbeddingModel == embedding_model,
                    Embeddings.SBERTEmbedding.cosine_distance(query_vector) < distance_threshold
                )
                .order_by(Embeddings.SBERTEmbedding.cosine_distance(query_vector))
                .limit(top_k)
                .all()
            )
        else:  # Default to OpenAI
            results = (
                self.session.query(
                    SplitDocument,
                    Embeddings.OpenAIEmbedding.cosine_distance(query_vector).label('distance')
                )
                .join(Embeddings, Embeddings.SplitID == SplitDocument.SplitID)
                .filter(
                    # Embeddings.EmbeddingModel == embedding_model,
                    Embeddings.OpenAIEmbedding.cosine_distance(query_vector) < distance_threshold
                )
                .order_by(Embeddings.OpenAIEmbedding.cosine_distance(query_vector))
                .limit(top_k)
                .all()
            )
        out: list[SplitWithSimilarityDistance] = []
        for split_doc, dist in results:
            logging.info(f"[Embeddings:{embedding_model}] SplitID={split_doc.SplitID} DocID={split_doc.DocID} Distance={dist}")
            out.append(
                SplitWithSimilarityDistance(
                    split_doc.SplitID,
                    split_doc.DocID,
                    split_doc.SplitContent,
                    split_doc.SplitStartOffset,
                    split_doc.SplitLength,
                    dist,
                )
            )
        return out

    def does_split_exist(self, doc_id, split_start_offset):
        """
        Check if a split document exists for the given doc_id and split_start_offset.
        """
        return self.session.query(SplitDocument).filter(
            SplitDocument.DocID == doc_id,
            SplitDocument.SplitStartOffset == split_start_offset
        ).first() is not None

    #This accomplishes getting the split content without doing a Join
    def get_split_content(self, split_id):
        # split = self.get_one_split(split_id)
        # Should we use doc_crud function here to retrieve the document?
        # doc = self.doc_crud.get_document_by_id(split.DocID)

        s = aliased(SplitDocument)
        d = aliased(Document)

        query = (
            select(
                s.SplitID,
                s.DocID,
                d.DateRead,
                func.substring(d.DocContent, s.SplitStartOffset + 1, s.SplitLength).label(
                    'SplitContent')
            )
            .join(d, s.DocID == d.DocID)
            .where(s.SplitID == split_id))
        # assert doc.DocContent[split.SplitStartOffset:split.SplitStartOffset +
        # split.SplitLength] == split.SplitContent, "Document content does not match split content"
        # return doc.DocContent[split.SplitStartOffset:split.SplitStartOffset + split.SplitLength]
        with Session(self.session.bind) as session:
            result = session.execute(query).fetchone()
            if result:
                return result.SplitContent
            return None