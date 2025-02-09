# Replace with your PostgreSQL connection details
import pg8000
from google.cloud.sql.connector import Connector, IPTypes
from sqlalchemy import create_engine

from database_access.docCrud import DatabaseConnection, DocumentCRUD


def build_engine():
    import os
    from dotenv import load_dotenv
    load_dotenv()

    #instance_connection_name = 'securitynowrag:us-west1:securitynowrag'
    PROJECT_ID = os.getenv('PROJECT_ID')
    REGION = os.getenv('REGION')
    INSTANCE_NAME = os.getenv('INSTANCE_NAME')
    DATABASE_NAME = os.getenv('DATABASE_NAME')
    USERNAME = os.getenv('USERNAME')
    PASSWORD = os.getenv('PASSWORD')

    connector = Connector(IPTypes.PUBLIC)
    instance_connection_name = 'securitynowrag:us-west1:securitynowrag'
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
        logging.info(f'Sorry failed to connect: {ex}')
        raise ex

    #DATABASE_URL = (f'postgresql+pg8000://{os.getenv("USERNAME")}:{os.getenv("PASSWORD")}@[
    # /cloudsql/{os.getenv("PROJECT_ID")}:{os.getenv("REGION")}:{os.getenv("INSTANCE_NAME")}]/{os.getenv("DATABASE_NAME")}')

    return engine

# Example usage
if __name__ == "__main__":
    db_connection = DatabaseConnection(build_engine())
    crud = DocumentCRUD(db_connection)

    # Add a document
    logging.info("Adding a document...")
    crud.add_document("Sample MetaData", "2023-10-01 10:00:00", "2023-10-01 10:00:00", "Sample Content")

    # Update the document
    logging.info("Updating the document...")
    crud.update_document(1, metadata="Updated MetaData")

    # Delete the document
    # logging.info("Deleting the document...")
    # crud.delete_document(1)

    logging.info("Operations completed.")