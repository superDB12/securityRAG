from database_access.docCrud import DocumentCRUD
from database_access.session_factory import SessionFactory
from grc_retriever import GRCRetriever
import logging

class Loader:
    def __init__(self):
        self.doc_crud = DocumentCRUD(SessionFactory())
        self.grc_retriever = GRCRetriever(self.doc_crud)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logging.info("Starting loader...")
    loader = Loader()
    loader.grc_retriever.load_current_year()
    # year_urls = loader.grc_retriever.get_year_urls("https://www.grc.com/securitynow.htm")
    # loader.grc_retriever.get_historical_docs(year_urls)
    logging.info("Loader finished")
