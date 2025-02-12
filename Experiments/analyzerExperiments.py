import os
import re
from datetime import datetime
import logging
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from database_access.docCrud import DocumentCRUD, DatabaseConnection
from database_access.engine_factory import EngineFactory

assert os.environ.get("OPENAI_API_KEY")!=None, "You need an OpenAI API Key"

class DocumentAnalyzer:
    def __init__(self):
        # self.embeddings = OpenAIEmbeddings()
        self.engine = EngineFactory().get_engine()
        self.doc_crud = DocumentCRUD(DatabaseConnection(self.engine))

    def extract_date(self, text):
        date_pattern = re.compile(r'\b('
                                  r'January|JANUARY|February|FEBRUARY|March|MARCH|April|APRIL|May'
                                  r'|MAY|June|JUNE|July|JULY|August|AUGUST'
                                  r'|September|SEPTEMBER|October|OCTOBER|November|NOVEMBER'
                                  r'|December|DECEMBER) \d{1,2}, \d{4}\b')
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

    def split_and_load_documents(self):
        documents = self.doc_crud.get_all_documents()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

        for doc in documents:
            if doc.DocContent is not None and doc.DocDate is not None and not doc.Processed:
                logging.info(f'Printing length of doc: {len(doc.DocContent)}')
                splits = text_splitter.split_text(doc.DocContent)
                for split in splits:
                    # vector = self.embeddings.embed_text(split)
                    self.doc_crud.add_split_document(doc.DocID, doc.MetaData, doc.DateRead,
                                                     doc.DocDate, split)
                    #Add load vector here or elsewhere?
                    logging.info(f"Added split document for DocID {doc.DocID}")
                self.doc_crud.update_document(doc.DocID, processed=True)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logging.info("Starting document analyzer...")
    analyzer = DocumentAnalyzer()
    analyzer.insert_doc_date()
    analyzer.split_and_load_documents()
    logging.info("Document analyzer finished")