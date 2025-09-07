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

#from .docCrud import Document, DocumentCRUD
from database_access.session_factory import Base

class Embeddings(Base):
    __tablename__ = 'embeddings'
    EmbeddingID = Column(Integer, primary_key=True, autoincrement=True)
    SplitID = Column(Integer, ForeignKey('split_documents.SplitID'))
    DocID = Column(Integer, ForeignKey('documents.DocID'))
    Embedding = Column(Vector(3072))
    EmbeddingModel = Column(String)

class EmbeddingsCRUD:
    def __init__(self, db_connection):
        self.session = db_connection.get_session()
        Base.metadata.create_all(db_connection.get_engine())

    def add_embedding(self, split_id, doc_id, embedding, embedding_model):
        new_embedding = Embeddings(SplitID=split_id,
                                   DocID=doc_id,
                                   Embedding=embedding,
                                   EmbeddingModel=embedding_model)
        self.session.add(new_embedding)
        self.session.commit()

    def get_embedding(self, embedding_id):
        embedding = self.session.query(Embeddings).filter(Embeddings.EmbeddingID == embedding_id).first()
        return embedding

    def update_embedding(self, embedding_id, split_id=None, doc_id=None, embedding=None, embedding_model=None):
        embedding_record = self.session.query(Embeddings).filter(Embeddings.EmbeddingID == embedding_id).first()
        if embedding_record:
            if split_id:
                embedding_record.SplitID = split_id
            if doc_id:
                embedding_record.DocID = doc_id
            if embedding:
                embedding_record.Embedding = embedding
            if embedding_model:
                embedding_record.EmbeddingModel = embedding_model
            self.session.commit()

    def delete_embedding(self, embedding_id):
        embedding_record = self.session.query(Embeddings).filter(Embeddings.EmbeddingID == embedding_id).first()
        if embedding_record:
            self.session.delete(embedding_record)
            self.session.commit()