import os
import re
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import RecursiveUrlLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Load the documents
class GRCRetriever:
    def __init__(self, doc_crud):
        self.doc_crud = doc_crud

    def load_current_year(self):
        logging.info("Loading docs...")
        anchor_rgx = r'<a\s+(?:[^>]*?\s+)?href="([^"]*(?=txt)[^"]*)"'
        file_path = "https://www.grc.com/securitynow.htm"
        loader = RecursiveUrlLoader(file_path,
                                    max_depth=4,
                                    link_regex=anchor_rgx,
                                    base_url="https://www.grc.com/sn")

        for doc in loader.lazy_load():

            if doc.metadata['source'].endswith('.txt') and '404 - File or directory not found.' not in doc.page_content:
                # logging.info(doc.metadata['source'])
                # logging.info(doc.page_content[:300])
                # logging.info("-------------------")
                (self.doc_crud.add_document(doc.metadata['source'], datetime.now(), None,
                 doc.page_content))

        logging.info("Done loading docs \n")

    def get_year_urls(self, main_url: str):
        response = requests.get(main_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        year_urls = []
        for link in soup.find_all('a', href=True):
            # logging.info(f'Found link: {link["href"]}')
            if re.match(r'/sn/past/\d{4}.htm', link['href']):
                year_urls.append('https://www.grc.com' + link['href'])
        for urls in year_urls:
            logging.info(f'Printing urls: {urls}')
        return year_urls

    #Function to load all docs for a given single year by passing url as parameter
    def get_historical_docs(self, year_urls):
        for url in year_urls:
            anchor_rgx = r'<a\s+(?:[^>]*?\s+)?href="([^"]*(?=txt)[^"]*)"'
            file_path = url
            loader = RecursiveUrlLoader(url,
                                        max_depth=4,
                                        link_regex=anchor_rgx,
                                        base_url="https://www.grc.com/sn")

            for doc in loader.lazy_load():

                if doc.metadata['source'].endswith('.txt'):
                    # logging.info(doc.metadata['source'])
                    # logging.info(doc.page_content[:300])
                    # logging.info("-------------------")
                    (self.doc_crud.add_document(doc.metadata['source'], datetime.now(), None,
                    doc.page_content))

        logging.info("Done loading docs \n")