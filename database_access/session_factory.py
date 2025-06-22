import os
from dotenv import load_dotenv
import pg8000
from google.cloud.sql.connector import Connector, IPTypes
from utils.secret_manager import get_secret # Added: for fetching secrets
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import logging

Base = declarative_base()

class SessionFactory:
    engine = None
    SessionMaker = None
    session = None

    def __init__(self):
        self.engine = self.create_engine()
        self.SessionMaker = sessionmaker(bind=self.engine)
        self.session = self.SessionMaker()

    def create_engine(self):
        try:
            load_dotenv() # Removed: .env no longer primary source for these
        except Exception as ex:
            logging.info(f'Sorry failed to load .env file: {ex}')
            raise ex

        PROJECT_ID = os.getenv('PROJECT_ID') # Still needed for Secret Manager & Cloud SQL
        REGION = os.getenv('REGION') # Still needed for Cloud SQL
        INSTANCE_NAME = os.getenv('INSTANCE_NAME') # Still needed for Cloud SQL

        if not PROJECT_ID:
            logging.error("PROJECT_ID environment variable not set.")
            raise ValueError("PROJECT_ID environment variable not set.")
        if not REGION:
            logging.error("REGION environment variable not set.")
            raise ValueError("REGION environment variable not set.")
        if not INSTANCE_NAME:
            logging.error("INSTANCE_NAME environment variable not set.")
            raise ValueError("INSTANCE_NAME environment variable not set.")

        try:
            DATABASE_NAME = get_secret(PROJECT_ID, 'DATABASE_NAME')
            DB_USERNAME = get_secret(PROJECT_ID, 'DB_USERNAME') # Changed from USERNAME to DB_USERNAME
            DB_PASSWORD = get_secret(PROJECT_ID, 'DB_PASSWORD') # Changed from PASSWORD to DB_PASSWORD
        except Exception as ex:
            logging.error(f'Failed to retrieve secrets from Google Secret Manager: {ex}')
            raise ex

        connector = Connector(IPTypes.PUBLIC)
        instance_connection_name = f'{PROJECT_ID}:{REGION}:{INSTANCE_NAME}'
        try:
            def getconn() -> pg8000.dbapi.Connection:
                conn: pg8000.dbapi.Connection = connector.connect(
                    instance_connection_name,
                    "pg8000",
                    user=DB_USERNAME, # Updated
                    password=DB_PASSWORD, # Updated
                    db=DATABASE_NAME, # Updated
                )
                return conn

            engine = create_engine(
                "postgresql+pg8000://",
                creator=getconn
            )

        #     logging.info('Connect fn connected')
        except Exception as ex:
            logging.info(f'Sorry failed to connect: {ex}')
            raise ex

        return engine

    def get_engine(self):
        return self.engine

    def get_session(self):
        return self.session