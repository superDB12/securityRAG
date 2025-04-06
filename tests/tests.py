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

    #TODO: Build this out to match tests and bring in QueryString from text files
    def test_for_expected_results_from_semantic_search(self, QueryString, ExpectedDocID,
                                                       ExpectedSplitID=None):
        doc_crud = DocumentCRUD(SessionFactory())
        test_phrase = "a tease for 160 critical patches from Microsoft?"
        max_splits = int(os.environ.get("MAX_SPLITS"))
        distance_threshold = float(os.environ.get("DIST_THRESHOLD"))

        embeddingEngine = OpenAIEmbeddings(model="text-embedding-3-large")
        query_vector = embeddingEngine.embed_query(test_phrase)

        split_crud = SplitCRUD(SessionFactory())
        vectors = split_crud.get_similar_vectors(query_vector, top_k=max_splits,
                                                 distance_threshold=distance_threshold)
        found_it = False

        for vector in vectors:
            doc_id = vector.DocID
            split_id = vector.SplitID
            if doc_id == ExpectedDocID and split_id == ExpectedSplitID:
                found_it = True
                break
        self.assertFalse(found_it, f'Expected results not found. QueryString: {test_phrase}, '
                                   f'ExpectedDocID: {ExpectedDocID}, ExpectedSplitID: {ExpectedSplitID}')

    def test_semantic_search(self):
        # prerequisite for this test is to run Analyzer to build all the splits and the vectors
        #define a test phrase
        # test_phrase = ("SHOW TEASE:  It's time for Security Now!.  Steve Gibson is here with a rundown of the, what is it, 160 critical patches Microsoft shipped last week on Patch Tuesday?  Microsoft's also forcing you to take Outlook.  GoDaddy is going to get much more serious about its hosting security.  And then, get ready, get your propeller hats on because there will be math.  We're going to brute force your one-time password authenticator.  Well, at least we'll talk about how hard or easy it would be to do.  It's going to be a fun episode, next on Security Now!."
        #                "LEO LAPORTE:  This is Security Now! with Steve Gibson, Episode 1009, recorded Tuesday, January 21st, 2025:  Attacking TOTP."
        #                "It's time for Security Now!, the show where we talk about security, privacy, protecting yourself and your loved ones on the great big vast Internet with this guy right here, our security in ch")
        # test_phrase = "What episode is 160 critical patches from Microsoft?"
        # test_phrase = "show tease"
        # test_phrase = "show tease: It's time for Security Now!"
        # test_phrase = "GoDaddy"
        # test_phrase = "Starwood Group"
        test_phrase = "STEVE:  That Starwood Group breach incident.  What we recall from that is that Marriott acquired the independent Starwood Group whose network security was a lackluster afterthought, if you can call it that.  You know, like way out of date.  They didn't bother to update, and there were, like, known, well-known problems.  But Marriott, the acquirer, never took the time to thoroughly vet what they were purchasing, and that lack of oversight over their purchase came back to bite them."
        max_splits = int(os.environ.get("MAX_SPLITS"))
        distance_threshold = float(os.environ.get("DIST_THRESHOLD"))

        embeddingEngine = OpenAIEmbeddings(model="text-embedding-3-large")
        query_vector = embeddingEngine.embed_query(test_phrase)

        # run our semantic search in SplitCrud with the predefined phrase
        split_crud = SplitCRUD(SessionFactory())
        vectors = split_crud.get_similar_vectors(query_vector, top_k=max_splits, distance_threshold=distance_threshold)

        # log the results
        logging.info(f"Found {len(vectors)} similar vectors.")
        self.assertFalse((len(vectors)==0),f"No similar vectors found for the query: '{test_phrase}'")

        for vector in vectors:
            logging.info("--------Begin Vector---------------")
            logging.info(f"Split ID: {vector.SplitID}, Doc ID: {vector.DocID}, Split Length: {vector.SplitLength}")
            # log the split content
            split_text = split_crud.get_split_content(vector.SplitID)
            logging.info(f"Split Content: {split_text}")
            logging.info("--------End Vector---------------")

        # assert if the one we expect isn't found


if __name__ == '__main__':
    unittest.main()
