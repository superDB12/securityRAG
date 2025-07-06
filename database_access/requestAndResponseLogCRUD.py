from datetime import datetime

from sqlalchemy import Column, Integer, String, DateTime, UniqueConstraint

from .session_factory import Base

class RequestAndResponseLog(Base):
    __tablename__ = 'request_and_response_log'
    QueryID = Column(Integer, primary_key=True, autoincrement=True)
    Query = Column(String)
    QueryResponse = Column(String)
    QueryResponseScore = Column(Integer)
    QueryResponseComments = Column(String, nullable=True)
    QueryConcept = Column(String)
    QueryConceptResponse = Column(String)
    QueryConceptResponseScore = Column(Integer)
    QueryConceptResponseComments = Column(String, nullable=True)
    Date = Column(DateTime)
    # SessionID = Column(String)
    # UserID = Column(String)
    # __table_args__ = (UniqueConstraint('RequestID', name='_requestid_sessionid_uc'),)

class RequestAndResponseLogCRUD:
    def __init__(self, db_connection):
        self.session = db_connection.get_session()
        Base.metadata.create_all(db_connection.get_engine())

    def add_query_and_response_log(self,
                                   query,
                                   query_response,
                                   query_response_score,
                                   query_response_comments,
                                   query_concept,
                                   query_concept_response,
                                   concept_score,
                                   concept_score_comments,
                                   date=datetime.now()):
        new_log = RequestAndResponseLog(Query=query, 
                                        QueryResponse=query_response,
                                        QueryResponseScore=query_response_score,
                                        QueryResponseComments = query_response_comments,
                                        QueryConcept=query_concept,
                                        QueryConceptResponse=query_concept_response,
                                        QueryConceptResponseScore=concept_score,
                                        QueryConceptResponseComments = concept_score_comments,
                                        Date=date
                                        )
        self.session.add(new_log)
        self.session.commit()

    def update_query_and_response_log(self,
                                      query_id,
                                      query,
                                      query_response,
                                      query_response_score,
                                      query_response_comments,
                                      query_concept,
                                      query_concept_response,
                                      concept_score,
                                      concept_score_comments,
                                      date=datetime.now()):
        if query_id is None:
            raise ValueError("Query ID must be provided for update.")
        log = self.session.query(RequestAndResponseLog).filter(RequestAndResponseLog.QueryID == query_id).first()
        if log:
            if query:
                log.Query = query
            if query_response:
                log.QueryResponse = query_response
            if query_response_score is not None:
                log.QueryResponseScore = query_response_score
            if query_concept:
                log.QueryConcept = query_concept
            if query_concept_response:
                log.QueryConceptResponse = query_concept_response
            if query_response_comment:
                log.QueryResponseComments = query_response_comments
            if concept_score is not None:
                log.QueryConceptResponseScore = concept_score
            if concept_score_comments:
                log.QueryConceptResponseComments = concept_score_comments
            if date:
                log.Date = date
            self.session.commit()

    def delete_request_and_response_log(self, request_id):
        log = self.session.query(RequestAndResponseLog).filter(RequestAndResponseLog.RequestID == request_id).first()
        if log:
            self.session.delete(log)
            self.session.commit()

    def get_all_request_and_response_logs(self):
        return self.session.query(RequestAndResponseLog).all()

    def get_request_and_response_log_by_id(self, request_id):
        return self.session.query(RequestAndResponseLog).filter(RequestAndResponseLog.RequestID == request_id).first()