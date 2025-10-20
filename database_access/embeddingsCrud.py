from datetime import datetime
import numpy as np
from dns.e164 import query
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
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
    # Embedding = Column(Vector(3072))
    OpenAIEmbedding = Column(Vector(3072))
    SBERTEmbedding = Column(Vector(384))
    # EmbeddingModel = Column(String)

# use the langchain openAI embedding model to generate embeddings
def calculate_openAI_embedding(text) -> np.ndarray:
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    embedding = embeddings.embed_query(text)
    return embedding

# use the sentence_transformers library to generate embeddings
def calculate_SBERT_embedding(text) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = model.encode(text).tolist()
    return embedding

class EmbeddingsCRUD:
    def __init__(self, db_connection):
        self.session = db_connection.get_session()
        Base.metadata.create_all(db_connection.get_engine())

    # def add_OpenAI_embedding(self, split_id, doc_id, text):
    #     embedding = _calculate_openAI_embedding(text)
    #     self.add_embedding(split_id, doc_id, embedding, "OpenAI")
    #
    # def add_SBERT_embedding(self, split_id, doc_id, text):
    #     embedding = _calculate_SBERT_embedding(text)
    #     # note the sbert embedding is
    #     self.add_embedding(split_id, doc_id, embedding, "sBert")

    #TBD Remove the Embedding and Embedding Model columns after this code is all working and the embeddings have been moved.
    def add_embedding(self, split_id, doc_id, split_text ):
        OpenAIEmbedding= calculate_openAI_embedding(split_text)
        SBERTEmbedding= calculate_SBERT_embedding(split_text)
        new_embedding = Embeddings(SplitID=split_id,
                                   DocID=doc_id,
                                   Embedding=None,   # deprecating this colum and moving to separate embedding columns
                                   OpenAIEmbedding=OpenAIEmbedding,
                                   SBERTEmbedding=SBERTEmbedding,
                                   EmbeddingModel = None # deprecating this column and moving to separate embedding columns
        )
        self.session.add(new_embedding)
        self.session.commit()

    def get_embedding(self, embedding_id):
        embedding = self.session.query(Embeddings).filter(Embeddings.EmbeddingID == embedding_id).first()
        return embedding

    def get_all_embeddings(self):
        # return self.session.query(Embeddings).all()
        return self.session.query(Embeddings)

    def update_embedding(self, embedding_id, split_id=None, doc_id=None, split_text = None, embedding=None, embedding_model=None):
        OpenAIEmbedding= calculate_openAI_embedding(split_text)
        SBERTEmbedding= calculate_SBERT_embedding(split_text)

        embedding_record = self.session.query(Embeddings).filter(Embeddings.EmbeddingID == embedding_id).first()
        if embedding_record:
            if split_id:
                embedding_record.SplitID = split_id
            if doc_id:
                embedding_record.DocID = doc_id
            if split_text:
                embedding_record.OpenAIEmbedding = OpenAIEmbedding
                embedding_record.SBERTEmbedding = SBERTEmbedding
            if embedding:
                embedding_record.Embedding = embedding
            if embedding_model:
                embedding_record.EmbeddingModel = embedding_model
            self.session.commit()

    def update_only_SBERT_embedding(self, embedding_id, split_id=None, doc_id=None, split_text = None, embedding=None, embedding_model=None):
        # OpenAIEmbedding= calculate_openAI_embedding(split_text)
        SBERTEmbedding= calculate_SBERT_embedding(split_text)

        embedding_record = self.session.query(Embeddings).filter(Embeddings.EmbeddingID == embedding_id).first()
        if embedding_record:
            if split_id:
                embedding_record.SplitID = split_id
            if doc_id:
                embedding_record.DocID = doc_id
            if split_text:
                # embedding_record.OpenAIEmbedding = OpenAIEmbedding
                embedding_record.SBERTEmbedding = SBERTEmbedding
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
