import unittest
from unittest.mock import patch, MagicMock
import os

# Set a dummy PROJECT_ID for testing if it's not already set
if "PROJECT_ID" not in os.environ:
    os.environ["PROJECT_ID"] = "test-project"
# Set a dummy REGION for testing if it's not already set
if "REGION" not in os.environ:
    os.environ["REGION"] = "test-region"
# Set a dummy INSTANCE_NAME for testing if it's not already set
if "INSTANCE_NAME" not in os.environ:
    os.environ["INSTANCE_NAME"] = "test-instance"

from retriever.concept_extractor import ConceptExtractor
from langchain_core.messages import AIMessage # For mocking LLM response


class TestConceptExtractor(unittest.TestCase):

    @patch('analyzer.concept_extractor.get_secret')
    @patch('analyzer.concept_extractor.init_chat_model')
    @patch('analyzer.concept_extractor.RequestAndResponseLogCRUD')
    @patch('analyzer.concept_extractor.SessionFactory')
    def setUp(self, MockSessionFactory, MockRequestAndResponseLogCRUD, MockInitChatModel, MockGetSecret):
        # Mock get_secret to return a dummy API key
        MockGetSecret.return_value = "fake_api_key"

        # Mock the language model
        self.mock_llm = MagicMock()
        MockInitChatModel.return_value = self.mock_llm

        # Mock CRUD and SessionFactory
        self.mock_logger_crud = MockRequestAndResponseLogCRUD.return_value
        MockSessionFactory.return_value = MagicMock()

        self.extractor = ConceptExtractor()

    # No changes to these tests, they use self.extractor which is correctly mocked.
    def test_extract_concept_buffer_overflow(self):
        question = "Can you tell me what a buffer overflow attack is in cybersecurity?"
        expected_concept = "buffer overflow attack in cybersecurity"
        self.mock_llm.invoke.return_value = AIMessage(content=expected_concept)
        result = self.extractor.extract_concept(question)
        self.assertEqual(result, expected_concept)
        self.mock_llm.invoke.assert_called_once()

    def test_extract_concept_http_vs_https(self):
        question = "What are the differences between HTTP and HTTPS?"
        expected_concept = "differences between HTTP and HTTPS"
        self.mock_llm.invoke.return_value = AIMessage(content=expected_concept)
        result = self.extractor.extract_concept(question)
        self.assertEqual(result, expected_concept)
        self.mock_llm.invoke.assert_called_once()

    def test_extract_concept_ram_vs_rom(self):
        question = "I’m trying to understand how RAM differs from ROM."
        expected_concept = "difference between RAM and ROM"
        self.mock_llm.invoke.return_value = AIMessage(content=expected_concept)
        result = self.extractor.extract_concept(question)
        self.assertEqual(result, expected_concept)
        self.mock_llm.invoke.assert_called_once()

    def test_extract_concept_java_garbage_collector(self):
        question = "How does the Java garbage collector work?"
        expected_concept = "java garbage collection"
        self.mock_llm.invoke.return_value = AIMessage(content=expected_concept)
        result = self.extractor.extract_concept(question)
        self.assertEqual(result, expected_concept)
        self.mock_llm.invoke.assert_called_once()

    @patch('analyzer.concept_extractor.SessionFactory')
    @patch('analyzer.concept_extractor.RequestAndResponseLogCRUD')
    @patch('analyzer.concept_extractor.init_chat_model')
    @patch('analyzer.concept_extractor.get_secret')
    def test_init_api_key_error_handling(self, MockGetSecret, MockInitChatModel, MockRequestAndResponseLogCRUD, MockSessionFactory):
        # Test that ConceptExtractor handles errors when fetching API key
        MockGetSecret.return_value = "fake_api_key" # Default for this test, will be overridden below
        MockGetSecret.side_effect = Exception("Failed to fetch secret")

        # Ensure dependent mocks are set up for this specific instantiation
        MockInitChatModel.return_value = MagicMock()
        MockRequestAndResponseLogCRUD.return_value = MagicMock()
        MockSessionFactory.return_value = MagicMock()

        # We expect logging.error to be called, but the object should still be created
        # and init_chat_model will likely fail later if the key is truly needed and not found by other means.
        # For this test, we're just checking that the constructor doesn't blow up immediately.
        with self.assertLogs(level='ERROR') as log:
            extractor = ConceptExtractor() # This instantiation was causing the problem
            self.assertIsNotNone(extractor)
            self.assertTrue(any("Error fetching OPENAI_API_KEY" in message for message in log.output))

    def test_llm_call_error_handling(self):
        question = "A question that causes an error."
        self.mock_llm.invoke.side_effect = Exception("LLM simulation error")

        with self.assertLogs(level='ERROR') as log:
            result = self.extractor.extract_concept(question)
            self.assertEqual(result, "Error extracting concept")
            self.assertTrue(any("Error during concept extraction LLM call" in message for message in log.output))

    def test_some_user_input(self):
        question = "What is the impact of quantum computing on cryptography?"
        expected_concept = "impact of quantum computing on cryptography"

        self.mock_llm.invoke.return_value = AIMessage(content=expected_concept)

        result = self.extractor.extract_concept(question)
        self.assertEqual(result, expected_concept)

if __name__ == '__main__':
    unittest.main()
