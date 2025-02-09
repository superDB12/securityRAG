from database_access.docCrud import DocumentCRUD, DatabaseConnection
from database_access.engine_factory import EngineFactory
from grc_retriever import GRCRetriever


class Loader:
    def __init__(self):
        self.engine = EngineFactory().get_engine()
        self.doc_crud = DocumentCRUD(DatabaseConnection(self.engine))
        self.grc_retriever = GRCRetriever(self.doc_crud)

if __name__ == "__main__":
    loader = Loader()
    loader.grc_retriever.load_docs()
