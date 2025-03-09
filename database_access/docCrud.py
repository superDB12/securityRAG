from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean
import os
import logging
from database_access.session_factory import Base

#Creating tables below
class Document(Base):
    __tablename__ = 'documents'
    DocID = Column(Integer, primary_key=True, autoincrement=True)
    EpisodeAirDate = Column(DateTime)
    PodcastTitle = Column(String)
    EpisodeNumber = Column(Integer)
    EpisodeTitle = Column(String)
    Hosts = Column(String)
    TranscriptionTextUrl = Column(String)
    DateRead = Column(DateTime)
    DocContent = Column(String)
    Processed = Column(Boolean, default=False)

class DocumentCRUD:
    def __init__(self, db_connection):
        self.session = db_connection.get_session()
        Base.metadata.create_all(db_connection.get_engine())

    def add_document(self, source_url, date_read, doc_date, doc_content,podcast_title=None,
                     episode_number=None, episode_title=None, hosts=None):
        doc = self.session.query(Document).filter(Document.TranscriptionTextUrl==source_url).first()
        if not doc:
            logging.info("Adding a document...")
            new_doc = Document(TranscriptionTextUrl=source_url, DateRead=date_read,
                               EpisodeAirDate=doc_date,
                               DocContent=doc_content, PodcastTitle=podcast_title,
                               EpisodeNumber=episode_number,EpisodeTitle=episode_title,
                               Hosts=hosts)
            self.session.add(new_doc)
            self.session.commit()
        else:
            logging.info("Document already exists")

    def update_document(self, doc_id, source_url=None, date_read=None, doc_date=None,
                        doc_content=None, processed=None, podcast_title=None,
                     episode_number=None, episode_title=None, hosts=None):
        doc = self.session.query(Document).filter(Document.DocID == doc_id).first()
        if doc:
            if source_url:
                doc.TranscriptionTextUrl = source_url
            if podcast_title:
                doc.PodcastTitle = podcast_title
            if episode_number:
                doc.EpisodeNumber = episode_number
            if episode_title:
                doc.EpisodeTitle = episode_title
            if hosts:
                doc.Hosts = hosts
            if date_read:
                doc.DateRead = date_read
            if doc_date:
                doc.EpisodeAirDate = doc_date
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
        return self.session.query(Document).filter(Document.EpisodeAirDate == None).all()

