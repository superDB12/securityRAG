import os
import unittest
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from database_access.docCrud import DocumentCRUD
from database_access.session_factory import SessionFactory
from database_access.splitCrud import SplitCRUD
import logging
from logging.config import dictConfig

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'DEBUG',
        'handlers': ['wsgi']
    }
})

class MyTestCase(unittest.TestCase):
    def test_get_split_by_splitID(self):
        split_crud = SplitCRUD(SessionFactory())
        a_split = split_crud.get_one_split(1)
        print(f'the size of the split is {a_split.SplitLength}')
        print(a_split)
        split_text = split_crud.get_split_content(a_split.SplitID)
        print(split_text)
        # self.assertEqual(True, False)  # add assertion here

    def test_chunking_size(self):
        split_crud = SplitCRUD(SessionFactory())
        doc_crud = DocumentCRUD(SessionFactory())
        doc_id = 1
        doc = doc_crud.get_document(doc_id)

        #get split setting from environment
        split_length = os.environ.get("SPLIT_LENGTH")
        split_overlap = os.environ.get("SPLIT_OVERLAP")

        # add_split_document(doc_id=1, split_start_offset=0, split_length=100, doc_vector=[0.1]*3072)

    def test_split_a_document(self):

        test_doc_id=1

        #get the document from the database
        doc_crud = DocumentCRUD(SessionFactory())
        doc = doc_crud.get_document_by_id(test_doc_id)
        #split the document
        if doc.DocContent is not None:
            logging.info(f'Printing length of doc: {len(doc.DocContent)}')
            split_start_offset = 0
            # episode_number = doc.EpisodeNumber
            # episode_title = doc.EpisodeTitle
            # episode_date = doc.EpisodeAirDate
            # episode_metadata = (f"Episode Number: {episode_number}, Episode Title: "
            #                     f"{episode_title}, Episode Date: {episode_date} : ")
            split_length = int(os.environ.get("SPLIT_LENGTH"))
            split_overlap = int(os.environ.get("SPLIT_OVERLAP"))
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=split_length,
                chunk_overlap=split_overlap)

            splits = text_splitter.split_text(doc.DocContent)
            for split in splits:
                actual_split_length = len(split)
                logging.info(f"Split length: {actual_split_length}")
                if actual_split_length <= split_length * .9:
                    logging.info(f"Split length is less than 90% of the split length, here is the split:")
                    logging.info(f"{split}")
                    logging.info("-----------------------")

    def test_semantic_search(self):
        # prerequisite for this test is to run Analyzer to build all the splits and the vectors
        #define a test phrase
        # run our semantic search in SplitCrud with the predefined phrase
        vectors = get_similar_vectors(self, query_vector, top_k=(int(os.environ.get("MAX_SPLITS"))),
                                distance_threshold=float(os.environ.get("DIST_THRESHOLD"))):

    # log the results
        # assert if the one we expect isn't found


if __name__ == '__main__':
    unittest.main()
