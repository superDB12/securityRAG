from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

Base = declarative_base()

class DatabaseConnection:
    def __init__(self, engine):
        self.engine = engine
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()

    def get_session(self):
        return self.session

class Document(Base):
    __tablename__ = 'documents'
    DocID = Column(Integer, primary_key=True, autoincrement=True)
    MetaData = Column(String)
    DateRead = Column(DateTime)
    DocDate = Column(DateTime)
    DocContent = Column(String)

class DocumentCRUD:
    def __init__(self, db_connection):
        self.session = db_connection.get_session()
        Base.metadata.create_all(db_connection.engine)

    def add_document(self, metadata, date_read, doc_date, doc_content):
        new_doc = Document(MetaData=metadata, DateRead=date_read, DocDate=doc_date, DocContent=doc_content)

        self.session.add(new_doc)
        self.session.commit()

    def update_document(self, doc_id, metadata=None, date_read=None, doc_date=None, doc_content=None):
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
            self.session.commit()

    def delete_document(self, doc_id):
        doc = self.session.query(Document).filter(Document.DocID == doc_id).first()
        if doc:
            self.session.delete(doc)
            self.session.commit()

# Replace with your PostgreSQL connection details
DATABASE_URL = "postgresql+psycopg2://<username>:<password>@<host>:<port>/<database>"

# Example usage
if __name__ == "__main__":
    db_connection = DatabaseConnection(DATABASE_URL)
    crud = DocumentCRUD(db_connection)
    # Add a document
    crud.add_document("Sample MetaData", "2023-10-01 10:00:00", "2023-10-01 10:00:00", "Sample Content")
    # Update a document
    crud.update_document(1, metadata="Updated MetaData")
    # Delete a document
    crud.delete_document(1)