import os
import re
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import RecursiveUrlLoader
import logging
from datetime import datetime # Ensure datetime is imported

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Load the documents
class GRCRetriever:
    def __init__(self, doc_crud):
        self.doc_crud = doc_crud

    # Copied from analyzer.py and modified
    def extract_date(self, text):
        date_pattern = re.compile(r'\b(January|February|March|April|May|June|July|August|September'
                            r'|October'
                    r'|November|December) \d{1,2}, \d{4}\b', re.IGNORECASE)
        match = date_pattern.search(text)
        if match:
            return datetime.strptime(match.group(), '%B %d, %Y')
        else:
            logging.warning("No date found in text.")
            return None

    def extract_series(self, text):
        series_pattern = re.compile(r'SERIES:\s+(.+)', re.IGNORECASE)
        match = series_pattern.search(text)
        if match:
            return match.group(1)
        # logging.warning("No series found in text, returning default 'Security Now!'.") # Already returns a default
        return "Security Now!"

    def extract_episode(self, text):
        episode_pattern = re.compile(r'EPISODE:\s+#(\d+)', re.IGNORECASE)
        match = episode_pattern.search(text)
        if match:
            return match.group(1)
        else:
            logging.warning("No episode found in text.")
            return None

    def extract_title(self, text):
        title_pattern = re.compile(r'TITLE:\s+(.+)', re.IGNORECASE)
        match = title_pattern.search(text)
        if match:
            # TODO: get rid of carriage returns from this field (from original comment)
            return match.group(1).replace('\r', '').replace('\n', ' ').strip()
        else:
            logging.warning("No title found in text.")
            return "Unknown"

    def extract_hosts(self, text):
        hosts_pattern = re.compile(r'HOSTS:\s+(.+)', re.IGNORECASE)
        match = hosts_pattern.search(text)
        if match:
            return match.group(1)
        else:
            logging.warning("No hosts found in text.")
            return "Unknown"

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
                header_data = doc.page_content[:500] # Extract header
                episode_number = self.extract_episode(header_data)

                if episode_number is None:
                    logging.warning(f"Could not extract episode number from {doc.metadata['source']}. Skipping.")
                    continue

                existing_doc = self.doc_crud.get_document_by_episode_number(episode_number)
                if existing_doc:
                    logging.info(f"Episode {episode_number} from {doc.metadata['source']} already exists. Skipping.")
                    continue

                doc_date = self.extract_date(header_data)
                doc_series = self.extract_series(header_data)
                doc_title = self.extract_title(header_data)
                doc_hosts = self.extract_hosts(header_data)

                self.doc_crud.add_document(
                    doc_url=doc.metadata['source'],
                    doc_date=doc_date,
                    doc_content=doc.page_content,
                    podcast_title=doc_series,
                    episode_number=episode_number,
                    episode_title=doc_title,
                    hosts=doc_hosts,
                    date_added=datetime.now()
                )
        logging.info("Done loading current year docs \n")

    def get_year_urls(self, main_url: str):
        response = requests.get(main_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        year_urls = []
        for link in soup.find_all('a', href=True):
            # logging.info(f'Found link: {link["href"]}')
            if re.match(r'/sn/past/\d{4}.htm', link['href']):
                year_urls.append('https://www.grc.com' + link['href'])
        # The following two lines were for debugging and are not strictly necessary
        # for urls in year_urls:
        #     logging.info(f'Printing urls: {urls}')
        return year_urls

    #Function to load all docs for a given single year by passing url as parameter
    def get_historical_docs(self, year_urls):
        for url in year_urls:
            anchor_rgx = r'<a\s+(?:[^>]*?\s+)?href="([^"]*(?=txt)[^"]*)"'
            # file_path = url # file_path is not used, url is used directly in RecursiveUrlLoader
            loader = RecursiveUrlLoader(url, # Changed file_path to url
                                        max_depth=4,
                                        link_regex=anchor_rgx,
                                        base_url="https://www.grc.com/sn")

            for doc in loader.lazy_load():
                if doc.metadata['source'].endswith('.txt'):
                    header_data = doc.page_content[:500] # Extract header
                    episode_number = self.extract_episode(header_data)

                    if episode_number is None:
                        logging.warning(f"Could not extract episode number from {doc.metadata['source']}. Skipping.")
                        continue

                    existing_doc = self.doc_crud.get_document_by_episode_number(episode_number)
                    if existing_doc:
                        logging.info(f"Episode {episode_number} from {doc.metadata['source']} already exists. Skipping.")
                        continue

                    doc_date = self.extract_date(header_data)
                    doc_series = self.extract_series(header_data)
                    doc_title = self.extract_title(header_data)
                    doc_hosts = self.extract_hosts(header_data)

                    self.doc_crud.add_document(
                        doc_url=doc.metadata['source'],
                        doc_date=doc_date,
                        doc_content=doc.page_content,
                        podcast_title=doc_series,
                        episode_number=episode_number,
                        episode_title=doc_title,
                        hosts=doc_hosts,
                        date_added=datetime.now()
                    )
        logging.info("Done loading historical docs \n")