import os

import pg8000
from google.cloud.sql.connector import Connector, IPTypes
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import create_engine

assert os.environ.get("OPENAI_API_KEY")!=None, "You need to set OPENAI_API_KEY."
assert os.environ.get("LANGCHAIN_TRACING_V2")!=None, "You need to set LANGCHAIN_TRACING_V2. "
assert os.environ.get("LANGCHAIN_API_KEY")!=None, "You need to set LANGCHAIN_API_KEY."

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"]="pr-smug-tutu-71"


documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
]


# load the doc
file_path = "./example_data/sn-1009.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

print(len(docs)) #107

# for page in docs:
#     print(page.metadata)
#     print(page.page_content[:100])


# split the text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(len(all_splits)) #516

# Embeddings
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}\n") #3072
print(vector_1[:10])


# Vector store
from langchain_postgres import PGVector

# instance_connection_name = 'securitynowrag:us-west1:securitynowrag'
PROJECT_ID = "securitynowrag"  # Your Google Cloud project ID
REGION = "us-west1"  # The region of your Cloud SQL instance
INSTANCE_NAME = "securitynowrag"  # The name of your Cloud SQL instance
DATABASE_NAME = "security-now-rag-db"  # The name of your database
USERNAME = "postgres"  # Your database username
PASSWORD = "68]IAgKdH=n}HX%m"  # Your database password

connector = Connector(IPTypes.PUBLIC)
instance_connection_name = f'{PROJECT_ID}:{REGION}:{INSTANCE_NAME}'

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

vector_store = PGVector(
    embeddings=embeddings,
    collection_name="my_docs",
    connection=engine,
)

ids = vector_store.add_documents(documents=all_splits)

results = vector_store.similarity_search(
    "What are CISA's efforts?"
)

print(results[0])
# results2 = vector_store.similarity_search("What did GoDaddy do?")
# print("GoDaddy did:" + " /n" )
# print(results2[0])

# async def async_search():
#     result = await vector_store.asimilarity_search("What did GoDaddy do?")
#     print(result[0])
#     return result
# async_search()



results = vector_store.similarity_search_with_score("What did GoDaddy do?")
doc, score = results[0]
print(f"Score: {score}\n")
print(doc)

embedding = embeddings.embed_query("What did GoDaddy do?")

results = vector_store.similarity_search_by_vector(embedding)
print(f"This is the embedding result \n")
print(results[0])