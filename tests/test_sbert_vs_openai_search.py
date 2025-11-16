import unittest
import os
from database_access import embeddingsCrud
import retriever

import logging
from dotenv import load_dotenv
from database_access.docCrud import DocumentCRUD
from database_access.session_factory import SessionFactory
from database_access.splitCrud import SplitCRUD
from retriever.retriever import DocumentSearcher
from utils.secret_manager import get_secret
from database_access.embeddingsCrud import EmbeddingsCRUD

class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.sessionFactory = SessionFactory()
        self.doc_crud = DocumentCRUD(self.sessionFactory)
        self.split_crud = SplitCRUD(self.sessionFactory)
        self.embeddings_crud = EmbeddingsCRUD(self.sessionFactory)
        self.document_searcher = DocumentSearcher()

    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    # python
    def test_read_user_query_file(self):
        cwd= os.getcwd()
        file_path = cwd + '/tests/test_data/user_queries.txt'
        print(file_path)
        lines = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                lines.append(line.rstrip('\n'))

        # optional sanity checks
        self.assertIsInstance(lines, list)


        for line in lines:
            # create embeddings for the query
            # sbert_query_embedding = embeddingsCrud.calculate_SBERT_embedding(line)
            # openai_query_embedding = embeddingsCrud.calculate_openAI_embedding(line)

            # make the cosign similarity queries
            sbert_splits = self.document_searcher.search_similar_splits_using_SBERT(line)
            openai_splits = self.document_searcher.search_similar_splits_using_OpenAI(line)
            for sbert_split in sbert_splits:
                print(sbert_split.SplitID)

            for openai_split in openai_splits:
                print(openai_split.SplitID)

if __name__ == '__main__':
    unittest.main()
