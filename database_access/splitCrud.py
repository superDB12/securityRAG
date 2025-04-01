from datetime import datetime

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

load_dotenv()
assert os.environ.get("MAX_SPLITS") is not None, "You have not set MAX_SPLITS."
assert os.environ.get("DIST_THRESHOLD") is not None, "You have not set DIST_THRESHOLD."

#TODO: Add metadata for the split documents

#Add a new table for split documents here or in a new file?
class SplitDocument(Base):
    __tablename__ = 'split_documents'
    SplitID = Column(Integer, primary_key=True, autoincrement=True)
    DocID = Column(Integer, ForeignKey('documents.DocID'))
    DateRead = Column(DateTime)
    # SplitContent = Column(String)
    SplitStartOffset = Column(Integer)
    SplitLength = Column(Integer)
    #Do we need VectorStored?
    # VectorStored = Column(Boolean, default=False)
    SplitVector = Column(Vector(3072))
    __table_args__ = (UniqueConstraint('DocID', 'SplitStartOffset', name='_docid_splitstartoffset_uc'),)

class SplitCRUD:
    def __init__(self, db_connection):
        self.session = db_connection.get_session()
        self.doc_crud = DocumentCRUD(db_connection)
        Base.metadata.create_all(db_connection.get_engine())

#TODO optimize to use offsets and only store the content in the documents table
    def add_split_document(self, doc_id, split_start_offset, split_length,
                           split_vector):
        existing_split = self.session.query(SplitDocument).filter(
            SplitDocument.DocID == doc_id,
            SplitDocument.SplitStartOffset == split_start_offset

        ).first()
        if not existing_split:
            new_split_doc: SplitDocument = SplitDocument(DocID=doc_id, DateRead=datetime.now(), SplitStartOffset=split_start_offset, SplitLength=split_length,
                                                         SplitVector=split_vector)
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

    def get_similar_vectors(self, query_vector, top_k=(int(os.environ.get("MAX_SPLITS"))), distance_threshold=float(os.environ.get("DIST_THRESHOLD"))):
        query_vector_size = len(query_vector)
        logging.info(f'Running get_similar_vectors with {query_vector_size} sized vector with top_k: {top_k}, distance threshold: {distance_threshold}')
        return self.session.query(SplitDocument).filter(
            SplitDocument.SplitVector.cosine_distance(query_vector) < distance_threshold
        ).order_by(
            SplitDocument.SplitVector.cosine_distance(query_vector)
        ).limit(top_k).all()

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