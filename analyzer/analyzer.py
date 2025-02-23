import os
import re
from datetime import datetime
import logging
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from database_access.docCrud import DocumentCRUD
from database_access.session_factory import SessionFactory
from database_access.splitCrud import SplitCRUD

load_dotenv()
assert os.environ.get("OPENAI_API_KEY") is not None, "You need an OpenAI API Key"
assert os.environ.get("SPLIT_LENGTH") is not None, "You need an SPLIT_LENGTH"
assert os.environ.get("SPLIT_OVERLAP") is not None, "You need an SPLIT_OVERLAP"

class DocumentAnalyzer:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        sessionFactory = SessionFactory()
        self.doc_crud = DocumentCRUD(sessionFactory)
        self.split_crud = SplitCRUD(sessionFactory)

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
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=int(os.environ.get("SPLIT_LENGTH")), chunk_overlap=int(os.environ.get("SPLIT_OVERLAP")))

        for doc in documents:
            if doc.DocContent is not None and doc.DocDate is not None and not doc.Processed:
                logging.info(f'Printing length of doc: {len(doc.DocContent)}')
                split_start_offset = 0
                # TODO: how does splitter know where to split
                splits = text_splitter.split_text(doc.DocContent)
                for split in splits:
                    vector = self.embeddings.embed_query(split)
                    logging.info(f"Generated vector of length {len(vector)} for split")
                    self.split_crud.add_split_document(doc.DocID, split_start_offset, len(split), vector)
                    logging.info(f"Added split document for DocID {doc.DocID}")
                    logging.info(f"Split start offset: {split_start_offset}, split length: {len(split)}")
                    split_start_offset = split_start_offset + len(split)
                self.doc_crud.update_document(doc.DocID, processed=True)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logging.info("Starting document analyzer...")
    analyzer = DocumentAnalyzer()
    analyzer.insert_doc_date()
    analyzer.load_splits_and_vectors()
    logging.info("Document analyzer finished")