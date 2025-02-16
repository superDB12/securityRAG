#TODO separate the docCRUD from the splitCRUD in this file

from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, \
    UniqueConstraint, Boolean, ARRAY, Float, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import logging
from pgvector.sqlalchemy import Vector

Base = declarative_base()

load_dotenv()
assert os.environ.get("MAX_SPLITS") is not None, "You have not set MAX_SPLITS.  Will default to 10"
assert os.environ.get("DIST_THRESHOLD") is not None, "You have not set DIST_THRESHOLD.  Will default to 0.7"


#Add a new table for split documents here or in a new file?
class SplitDocument(Base):
    __tablename__ = 'split_documents'
    SplitID = Column(Integer, primary_key=True, autoincrement=True)
    DocID = Column(Integer, ForeignKey('documents.DocID'))
    DateRead = Column(DateTime)
    SplitContent = Column(String)
    #Do we need VectorStored?
    VectorStored = Column(Boolean, default=False)
    SplitVector = Column(Vector(3072))
    __table_args__ = (UniqueConstraint('DocID', 'SplitContent', name='_docid_splitcontent_uc'),)

class SplitCRUD:
    def __init__(self, db_connection):
        self.session = db_connection.get_session()
        Base.metadata.create_all(db_connection.engine)

#TODO optimize to use offsets and only store the content in the documents table
    def add_split_document(self, doc_id, doc_content,
        doc_vector, vector_stored=False):
        existing_split = self.session.query(SplitDocument).filter(
            SplitDocument.DocID == doc_id,
            SplitDocument.SplitContent == doc_content

        ).first()
        if not existing_split:
            new_split_doc: SplitDocument = SplitDocument(DocID=doc_id, DateRead=datetime.now(),
                                          SplitContent=doc_content,
                                          SplitVector=doc_vector, VectorStored=vector_stored)
            self.session.add(new_split_doc)
            self.session.commit()
        else:
            logging.info(f"Duplicate split found for DocID {doc_id}, skipping insertion.")

    def update_split_document(self, split_id, doc_id=None, doc_content=None, vector_stored=None):
        split = self.session.query(SplitDocument).filter(SplitDocument.SplitID == split_id).first()
        if split:
            if doc_id:
                split.DocID = doc_id
            if doc_content:
                split.SplitContent = doc_content
            if vector_stored:
                split.VectorStored = vector_stored
            self.session.commit()

    def get_one_split(self, split_id):
        return self.session.query(SplitDocument).filter(SplitDocument.SplitID == split_id).first()

    def get_all_splits(self):
        return self.session.query(SplitDocument).all()

#TODO refactor to add the top_k and cosine distance into a .env file / GCLoud config thing
    def get_similar_vectors(self, query_vector, top_k=(int(os.environ.get("MAX_SPLITS"))), distance_threshold=float(os.environ.get("DIST_THRESHOLD"))):
        return self.session.query(SplitDocument).filter(
            SplitDocument.SplitVector.cosine_distance(query_vector) < distance_threshold
        ).order_by(
            SplitDocument.SplitVector.cosine_distance(query_vector)
        ).limit(top_k).all()

