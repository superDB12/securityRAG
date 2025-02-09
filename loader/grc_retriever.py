import os
from datetime import datetime
from langchain_community.document_loaders import RecursiveUrlLoader

# Load the documents
class GRCRetriever:
    def __init__(self, doc_crud):
        self.doc_crud = doc_crud

    def load_docs(self):
        print("Loading docs...")
        anchor_rgx = r'<a\s+(?:[^>]*?\s+)?href="([^"]*(?=txt)[^"]*)"'
        file_path = "https://www.grc.com/securitynow.htm"
        loader = RecursiveUrlLoader(file_path,
                                    max_depth=4,
                                    link_regex=anchor_rgx,
                                    base_url="https://www.grc.com/sn")

        for doc in loader.lazy_load():

            if doc.metadata['source'].endswith('.txt'):
                # print(doc.metadata['source'])
                # print(doc.page_content[:300])
                # print("-------------------")
                (self.doc_crud.add_document(doc.metadata['source'], datetime.now(), None,
                 doc.page_content))

        print("Done loading docs \n")
