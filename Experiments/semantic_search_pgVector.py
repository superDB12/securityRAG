import os

import pg8000
from google.cloud.sql.connector import Connector, IPTypes
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import create_engine
# Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

assert os.environ.get("OPENAI_API_KEY")!=None, "You need to set OPENAI_API_KEY."
assert os.environ.get("LANGCHAIN_TRACING_V2")!=None, "You need to set LANGCHAIN_TRACING_V2. "
assert os.environ.get("LANGCHAIN_API_KEY")!=None, "You need to set LANGCHAIN_API_KEY."
assert os.environ.get('DB_PASSWORD')!=None, "You need to set DB_PASSWORD."

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"]="pr-smug-tutu-71"


class SemanticSearchExperiment:
    vector_store = None
    PROJECT_ID = "securitynowrag"  # Your Google Cloud project ID
    REGION = "us-west1"  # The region of your Cloud SQL instance
    INSTANCE_NAME = "securitynowrag"  # The name of your Cloud SQL instance
    DATABASE_NAME = "security-now-rag-db"  # The name of your database
    USERNAME = "postgres"  # Your database username
    PASSWORD = os.environ.get('DB_PASSWORD')
    # PASSWORD =  "68]IAgKdH=n}HX%m"
    connector = Connector(IPTypes.PUBLIC)
    instance_connection_name = f'{PROJECT_ID}:{REGION}:{INSTANCE_NAME}'

    def initialize_data(self):
        # load the doc
        file_path = "./sn-1009.pdf"
        loader = PyPDFLoader(file_path)

        docs = loader.load()

        # split the text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        all_splits = text_splitter.split_documents(docs)

        print(len(all_splits)) #516
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        # Vector store

        def getconn() -> pg8000.dbapi.Connection:
            conn: pg8000.dbapi.Connection = self.connector.connect(
                self.instance_connection_name,
                "pg8000",
                user=self.USERNAME,
                password=self.PASSWORD,
                db=self.DATABASE_NAME,
            )
            return conn

        engine = create_engine(
            "postgresql+pg8000://",
            creator=getconn
        )

        self.vector_store = PGVector(
            embeddings=embeddings,
            collection_name="my_docs",
            connection=engine,
        )

        # This adds and vectorizes splits to the Vector Store
        ids = self.vector_store.add_documents(documents=all_splits)

        return ids

    def semantic_search(self, phrase):
        search_results = self.vector_store.similarity_search(phrase)

        print(search_results[0])

        return search_results
