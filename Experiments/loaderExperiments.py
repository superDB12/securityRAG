import os
assert os.environ.get("OPENAI_API_KEY")!=None, "You need an OpenAI API Key"
assert os.environ.get("LANGCHAIN_API_KEY")!=None, "You need to set LANGCHAIN_API_KEY"
import logging
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = InMemoryVectorStore(embeddings)

# Load the documents
def load_docs():
    logging.info("Loading docs...")
    from langchain_community.document_loaders import RecursiveUrlLoader
    anchor_rgx = r'<a\s+(?:[^>]*?\s+)?href="([^"]*(?=txt)[^"]*)"'
    file_path = "https://www.grc.com/securitynow.htm"
    loader = RecursiveUrlLoader(file_path,
                                max_depth=4,
                                link_regex=anchor_rgx,
                                base_url="https://www.grc.com/sn")

    pages = []
    for doc in loader.lazy_load():

        if doc.metadata['source'].endswith('.txt'):
            logging.info(doc.metadata['source'])
            logging.info(doc.page_content[:300])
            logging.info("-------------------")
            pages.append(doc)

        if len(pages) >= 10:
            # do some paged operation, e.g.
            # index.upsert(page)
            logging.info("We Loaded 10 Docs")
            break
            #pages = []

    logging.info(len(pages))
    logging.info("Done loading docs \n")

    return pages

if __name__ == "__main__":
    pages = load_docs()

    # Split the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(pages)

    # Index the documents
    ids = vector_store.add_documents(documents=all_splits)

    # Search for a query
    results = vector_store.similarity_search_with_score("What was the main topic of Security Now's latest episode?")
    doc, score = results[0]
    logging.info(f"Score: {score}\n")
    logging.info(doc.metadata['source'])
    logging.info(doc.page_content[:300])


