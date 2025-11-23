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
        file_path = cwd + '/test_data/user_queries.txt'
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
            # for sbert_split in sbert_splits:
                # print(sbert_split.SplitID)

            # for openai_split in openai_splits:
                # print(openai_split.SplitID)

            self.show_cossimilarityd_scores(sbert_splits, openai_splits)

            print("----")
            self.compare_embedding_lists(sbert_splits, openai_splits)

    #         TODO: hand these results back to Open Ai to generate an answer and compare
    #          qualitatively.  Does having more splits create a more accurate answer?  Which has
    #          better density of information- is the detail equal?  How do we score the density
    #          of technical information in the answer?

    def show_cossimilarityd_scores(self, SBERTList, OpenAIList):
        print("list of SBERT cosine similarity scores:")
        for sbert_item in SBERTList:
            print("sbert_item SplitID:", sbert_item.SplitID, "sbert_item CosineSimilarity:",
                  sbert_item.SplitCosignDistance)

        print("list of OpenAI cosine similarity scores:")
        for openai_item in OpenAIList:
            print("openai_item SplitID:", openai_item.SplitID, "openai_item CosineSimilarity:",
                  openai_item.SplitCosignDistance)

    def compare_embedding_lists(self, SBERTList, OpenAIList):
        """Compare two lists of embeddings and return similarity score."""
        if len(SBERTList) == len(OpenAIList):
            print("Lists are of equal length.")
        set1 = {item.SplitID for item in SBERTList}
        set2 = {item.SplitID for item in OpenAIList}

        unique_items_in_SBERT = set1 - set2
        unique_items_in_OpenAI = set2 - set1

        print("Values unique to SBERT:", unique_items_in_SBERT)
        for item in SBERTList:
            if item.SplitID in unique_items_in_SBERT:
                print(item)

        print("Values unique to OpenAI:", unique_items_in_OpenAI)
        for item in OpenAIList:
            if item.SplitID in unique_items_in_OpenAI:
                print(item)



if __name__ == '__main__':
    unittest.main()
