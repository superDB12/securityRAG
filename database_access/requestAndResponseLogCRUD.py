from sqlalchemy import Column, Integer, String, DateTime, UniqueConstraint

from .session_factory import Base

class RequestAndResponseLog(Base):
    __tablename__ = 'request_and_response_log'
    RequestID = Column(Integer, primary_key=True, autoincrement=True)
    Request = Column(String)
    Response = Column(String)
    Date = Column(DateTime)
    # SessionID = Column(String)
    # UserID = Column(String)
    UserScore = Column(Integer)
    UserComment = Column(String)
    # __table_args__ = (UniqueConstraint('RequestID', name='_requestid_sessionid_uc'),)

class RequestAndResponseLogCRUD:
    def __init__(self, db_connection):
        self.session = db_connection.get_session()
        Base.metadata.create_all(db_connection.get_engine())

    def add_request_and_response_log(self, request, response, date, score=None, comment=None):
        new_log = RequestAndResponseLog(Request=request, Response=response, Date=date, UserScore=score, UserComment=comment)
        self.session.add(new_log)
        self.session.commit()

    def update_request_and_response_log(self, request_id, request=None, response=None, date=None, score=None, comment=None):
        log = self.session.query(RequestAndResponseLog).filter(RequestAndResponseLog.RequestID == request_id).first()
        if log:
            if request:
                log.Request = request
            if response:
                log.Response = response
            if date:
                log.Date = date
            if score:
                log.UserScore = score
            if comment:
                log.UserComment = comment
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