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
        else:
            assert False, "No date found"


    def extract_series(self, text):
        series_pattern = re.compile(r'SERIES:\s+(.+)', re.IGNORECASE)
        match = series_pattern.search(text)
        if match:
            return match.group(1)
        return "Security Now!"

    def extract_episode(self, text):
        episode_pattern = re.compile(r'EPISODE:\s+#(\d+)', re.IGNORECASE)
        match = episode_pattern.search(text)
        if match:
            return match.group(1)
        else:
            assert False, "No episode found"

    # TODO: get rid of carriage returns from this field
    def extract_title(self, text):
        title_pattern = re.compile(r'TITLE:\s+(.+)', re.IGNORECASE)
        match = title_pattern.search(text)
        if match:
            return match.group(1)
        else:
            assert False, "No title found"

    def extract_hosts(self, text):
        hosts_pattern = re.compile(r'HOSTS:\s+(.+)', re.IGNORECASE)
        match = hosts_pattern.search(text)
        if match:
            return match.group(1)
        else:
            assert False, "No hosts found"

    def insert_episode_data(self):
        documents = self.doc_crud.get_documents_with_null_doc_date()
        for doc in documents:
            header_data = doc.DocContent[:400]
            doc_date = self.extract_date(header_data)
            doc_series = self.extract_series(header_data)
            doc_title = self.extract_title(header_data)
            doc_episode = self.extract_episode(header_data)
            doc_hosts = self.extract_hosts(header_data)

            self.doc_crud.update_document(doc.DocID, doc_date=doc_date, podcast_title=doc_series, episode_number=doc_episode, episode_title=doc_title, hosts=doc_hosts)
            logging.info(f"Updated document {doc.DocID} with date {doc_date}")


    def load_splits_and_vectors (self):
        documents = self.doc_crud.get_all_documents()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=int(os.environ.get("SPLIT_LENGTH")), chunk_overlap=int(os.environ.get("SPLIT_OVERLAP")))

        for doc in documents:
            if doc.DocContent is not None and doc.EpisodeAirDate is not None and not doc.Processed:
                logging.info(f'Printing length of doc: {len(doc.DocContent)}')
                split_start_offset = 0
                episode_number = doc.EpisodeNumber
                episode_title = doc.EpisodeTitle
                episode_date = doc.EpisodeAirDate
                episode_metadata= (f"Episode Number: {episode_number}, Episode Title: "
                                   f"{episode_title}, Episode Date: {episode_date} : ")
                splits = text_splitter.split_text(doc.DocContent)
                for split in splits:
                    vector = self.embeddings.embed_query(episode_metadata + split)
                    #TODO: ensure that the metadata is actually being appended to the split content
                    logging.info(f"Episode metadata: {episode_metadata}")
                    logging.info(f"Split content: {split}")
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
    analyzer.insert_episode_data()
    analyzer.load_splits_and_vectors()
    logging.info("Document analyzer finished")