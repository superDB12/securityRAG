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

import logging
from logging.config import dictConfig
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
    SplitVector = Column(Vector(3072))

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
                           split_vector, SplitContent=None):
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
                "SplitVector": split_vector
            }
            if DEBUG_SPLIT_CONTENT_ENABLED and SplitContent is not None:
                split_doc_args["SplitContent"] = SplitContent

            new_split_doc: SplitDocument = SplitDocument(**split_doc_args)
            self.session.add(new_split_doc)
            self.session.commit()
        else:
            logging.info(f"Duplicate split found for DocID {doc_id}, skipping insertion.")

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
        # Use OpenAIEmbeddings to convert the query string to a vector
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        query_vector = embeddings.embed_query(query_string)
        logging.info(f'Running get_similar_splits_from_string with query: {query_string}')
        result = self.get_similar_splits_from_vector(query_vector, top_k, distance_threshold)
        return result

    def get_similar_splits_from_vector(self, query_vector, top_k=(int(os.environ.get("MAX_SPLITS"))), distance_threshold=float(os.environ.get("DIST_THRESHOLD"))) -> list[SplitWithSimilarityDistance]:
        query_vector_size = len(query_vector)
        # David and John confirmed we do not need to normalize the vectors
        # normalized_query_vector = query_vector
        # normalized_query_vector = self.normalize_split_vectors(query_vector)
        logging.info(f'Running get_similar_vectors with {query_vector_size} sized vector with top_k: {top_k}, distance threshold: {distance_threshold}')
        #TODO assign query results to variable and log it to make it easier to debug
        results = self.session.query(
            SplitDocument,
            SplitDocument.SplitVector.cosine_distance(query_vector).label('distance')
        ).filter(
            SplitDocument.SplitVector.cosine_distance(query_vector) < distance_threshold
        ).order_by(
            SplitDocument.SplitVector.cosine_distance(query_vector)
        ).limit(top_k).all()

        get_similar_vector_query_results: list[SplitWithSimilarityDistance] = []
        for result in results:
            split_query_result = {}
            logging.info(f"SplitID: {result[0].SplitID}, DocID: {result[0].DocID}, Distance:"
                         f" {result[1]}")
            # logging.info( f"\nNormalized Query:{query_vector}"
            #               f"\nStored Vector...: {result[0].SplitVector}")
            get_similar_vector_query_result = SplitWithSimilarityDistance(result[0].SplitID, result[0].DocID, result[0].SplitContent, result[0].SplitStartOffset, result[0].SplitLength, result[1])
            # split_distance_query_result['SpitID'] = result[0].SplitID
            # split_distance_query_result['DocID'] = result[0].DocID
            # split_distance_query_result['SplitContent'] = result[0].SplitContent
            # split_distance_query_result['SplitStartOffset'] = result[0].SplitStartOffset
            # split_distance_query_result['SplitLength'] = result[0].SplitLength
            # split_distance_query_result['SplitCosignDistance'] = result[1]
            # split_doc_array.append(result[0])
            get_similar_vector_query_results.append(get_similar_vector_query_result)
        return get_similar_vector_query_results

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
                    'SplitContent'),
                s.SplitVector
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