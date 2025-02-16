import os
import re
from datetime import datetime
import logging
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from database_access.docCrud import DocumentCRUD
from database_access.session_factory import SessionFactory

load_dotenv()
assert os.environ.get("OPENAI_API_KEY") is not None, "You need an OpenAI API Key"

class DocumentAnalyzer:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.doc_crud = DocumentCRUD(SessionFactory())

    def extract_date(self, text):
        date_pattern = re.compile(r'\b(January|February|March|April|May|June|July|August|September'
                            r'|October'
                    r'|November|December) \d{1,2}, \d{4}\b', re.IGNORECASE)
        match = date_pattern.search(text)
        if match:
            return datetime.strptime(match.group(), '%B %d, %Y')
        return None

    def insert_doc_date(self):
        documents = self.doc_crud.get_documents_with_null_doc_date()
        for doc in documents:
            doc_date = self.extract_date(doc.DocContent)
            if doc_date:
                self.doc_crud.update_document(doc.DocID, doc_date=doc_date)
                logging.info(f"Updated document {doc.DocID} with date {doc_date}")
            else:
                logging.info(f"No date found in document {doc.DocID}")

    def load_splits_and_vectors (self):
        documents = self.doc_crud.get_all_documents()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

        for doc in documents:
            if doc.DocContent is not None and doc.DocDate is not None and not doc.Processed:
                logging.info(f'Printing length of doc: {len(doc.DocContent)}')
                splits = text_splitter.split_text(doc.DocContent)
                for split in splits:
                    vector = self.embeddings.embed_query(split)
                    logging.info(f"Generated vector of length {len(vector)} for split")
                    self.doc_crud.add_split_document(doc.DocID, split, vector,
                                                     vector_stored=True)
                    logging.info(f"Added split document for DocID {doc.DocID}")
                self.doc_crud.update_document(doc.DocID, processed=True)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logging.info("Starting document analyzer...")
    analyzer = DocumentAnalyzer()
    analyzer.insert_doc_date()
    analyzer.load_splits_and_vectors()
    logging.info("Document analyzer finished")