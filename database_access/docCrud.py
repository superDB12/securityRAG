from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, \
    UniqueConstraint, Boolean, ARRAY, Float, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import logging
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class DatabaseConnection:
    def __init__(self, engine):
        self.engine = engine
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()

    def get_session(self):
        return self.session

#Creating tables below
class Document(Base):
    __tablename__ = 'documents'
    DocID = Column(Integer, primary_key=True, autoincrement=True)
    MetaData = Column(String)
    DateRead = Column(DateTime)
    DocDate = Column(DateTime)
    DocContent = Column(String)
    Processed = Column(Boolean, default=False)

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

class DocumentCRUD:
    def __init__(self, db_connection):
        self.session = db_connection.get_session()
        Base.metadata.create_all(db_connection.engine)

    def add_document(self, metadata, date_read, doc_date, doc_content):
        doc = self.session.query(Document).filter(Document.MetaData==metadata).first()
        if not doc:
            logging.info("Adding a document...")
            new_doc = Document(MetaData=metadata, DateRead=date_read, DocDate=doc_date, DocContent=doc_content)
            self.session.add(new_doc)
            self.session.commit()
        else:
            logging.info("Document already exists")

    def update_document(self, doc_id, metadata=None, date_read=None, doc_date=None,
                        doc_content=None, processed=None):
        doc = self.session.query(Document).filter(Document.DocID == doc_id).first()
        if doc:
            if metadata:
                doc.MetaData = metadata
            if date_read:
                doc.DateRead = date_read
            if doc_date:
                doc.DocDate = doc_date
            if doc_content:
                doc.DocContent = doc_content
            if processed:
                doc.Processed = processed
            self.session.commit()

    def delete_document(self, doc_id):
        doc = self.session.query(Document).filter(Document.DocID == doc_id).first()
        if doc:
            self.session.delete(doc)
            self.session.commit()

    def get_all_documents(self):
        return self.session.query(Document).all()

    def get_documents_with_null_doc_date(self):
        return self.session.query(Document).filter(Document.DocDate == None).all()

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

    def get_similar_vectors(self, query_vector, top_k=10):
        return self.session.query(SplitDocument).order_by(
            SplitDocument.SplitVector.cosine_distance(query_vector)
        ).limit(top_k).all()

