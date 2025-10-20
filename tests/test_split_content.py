# Unit test for verifying split content retrieval from the database.
# developed because of a nasty bug where the starting offset was miscalculated
# resulting in incorrect split content being stored and retrieved.
#
# Do NOT assume that splits are contiguous in the original document content.
# Langchain's splits were observed to not be continuous in some cases.


import os
import unittest

import logging
from dotenv import load_dotenv
from database_access.docCrud import DocumentCRUD
from database_access.session_factory import SessionFactory
from database_access.splitCrud import SplitCRUD
from utils.secret_manager import get_secret
from database_access.embeddingsCrud import EmbeddingsCRUD

class TestSplitContent(unittest.TestCase):
    def setUp(self):
        self.sessionFactory = SessionFactory()
        self.doc_crud = DocumentCRUD(self.sessionFactory)
        self.split_crud = SplitCRUD(self.sessionFactory)
        self.embeddings_crud = EmbeddingsCRUD(self.sessionFactory)

    def test_split_content_retrieval(self):
        # Assuming there is at least one split in the database for testing
        splits = self.split_crud.get_all_splits()
        self.assertGreater(len(splits), 0, "No splits found in the database for testing.")

        for split in splits:
            split_content_from_offset = self.split_crud.get_split_content(split.SplitID)
            split_content = split.SplitContent
            if split_content != split_content_from_offset:
                logging.warning(f"Mismatch in split content for SplitID {split.SplitID}")
            self.assertEqual(split_content_from_offset, split_content)
            #
            # self.assertIsNotNone(split_content, f"Split content for SplitID {split.SplitID} is None.")
            # self.assertIsInstance(split_content, str, f"Split content for SplitID {split.SplitID} is not a string.")
            # self.assertGreater(len(split_content), 0, f"Split content for SplitID {split.SplitID} is empty.")



if __name__== "__main__":
    unittest.main()