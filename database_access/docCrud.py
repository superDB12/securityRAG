from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean
import os
import logging
from database_access.session_factory import Base


#Creating tables below
class Document(Base):
    __tablename__ = 'documents'
    DocID = Column(Integer, primary_key=True, autoincrement=True)
    MetaData = Column(String)
    DateRead = Column(DateTime)
    DocDate = Column(DateTime)
    DocContent = Column(String)
    Processed = Column(Boolean, default=False)

class DocumentCRUD:
    def __init__(self, db_connection):
        self.session = db_connection.get_session()
        Base.metadata.create_all(db_connection.get_engine())

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

    def get_document_by_id(self, doc_id):
        return self.session.query(Document).filter(Document.DocID == doc_id).first()

    def get_documents_with_null_doc_date(self):
        return self.session.query(Document).filter(Document.DocDate == None).all()

