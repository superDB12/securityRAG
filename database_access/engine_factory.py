import os
from dotenv import load_dotenv
import pg8000
from google.cloud.sql.connector import Connector, IPTypes
from sqlalchemy import create_engine

class EngineFactory:
    engine = None
    def __init__(self):
        self.engine = self.create_engine()

    def create_engine(self):
        try:
            load_dotenv()
        except Exception as ex:
            print(f'Sorry failed to load .env file: {ex}')
            raise ex

        # instance_connection_name = 'securitynowrag:us-west1:securitynowrag'
        PROJECT_ID = os.getenv('PROJECT_ID')
        REGION = os.getenv('REGION')
        INSTANCE_NAME = os.getenv('INSTANCE_NAME')
        DATABASE_NAME = os.getenv('DATABASE_NAME')
        USERNAME = os.getenv('USERNAME')
        PASSWORD = os.getenv('PASSWORD')

        connector = Connector(IPTypes.PUBLIC)
        instance_connection_name = f'{PROJECT_ID}:{REGION}:{INSTANCE_NAME}'
        try:
            def getconn() -> pg8000.dbapi.Connection:
                conn: pg8000.dbapi.Connection = connector.connect(
                    instance_connection_name,
                    "pg8000",
                    user=USERNAME,
                    password=PASSWORD,
                    db=DATABASE_NAME,
                )
                return conn

            engine = create_engine(
                "postgresql+pg8000://",
                creator=getconn
            )

        #     logging.info('Connect fn connected')
        except Exception as ex:
            print(f'Sorry failed to connect: {ex}')
            raise ex

        return engine

    def get_engine(self):
        return self.engine