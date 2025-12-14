import unittest
import os
from fileinput import close
from pathlib import Path

import json

from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

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

# class MyTestCase(unittest.TestCase):
class SBERT_vs_OpenAI_evaluator():
    def setUp(self):
        self.sessionFactory = SessionFactory()
        self.doc_crud = DocumentCRUD(self.sessionFactory)
        self.split_crud = SplitCRUD(self.sessionFactory)
        self.embeddings_crud = EmbeddingsCRUD(self.sessionFactory)
        self.document_searcher = DocumentSearcher()
        self.html_document_number = 1

    def test_read_user_query_file(self):
        # cwd= os.getcwd()
        cwd= Path(__file__).resolve().parent
        file_path = f'{cwd}/test_data/user_queries.txt'
        print(f'Loading test queries from {file_path}')
        #/Users/johnfunk/Documents/code/securityRAG/tests/test_data/user_queries.txt
        lines = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                lines.append(line.rstrip('\n'))

        # optional sanity checks
        #self.assertIsInstance(lines, list)


        for line in lines:
            # create embeddings for the query
            # sbert_query_embedding = embeddingsCrud.calculate_SBERT_embedding(line)
            # openai_query_embedding = embeddingsCrud.calculate_openAI_embedding(line)

            # make the Cosine similarity queries
            sbert_splits = self.document_searcher.search_similar_splits_using_SBERT(line)
            openai_splits = self.document_searcher.search_similar_splits_using_OpenAI(line)
            # for sbert_split in sbert_splits:
                # print(sbert_split.SplitID)

            # for openai_split in openai_splits:
                # print(openai_split.SplitID)

            # self.show_cossimilarityd_scores(sbert_splits, openai_splits)

            # print("----")
            # self.compare_embedding_lists(sbert_splits, openai_splits)

            compare_file = self.format_results_as_html_table(line, sbert_splits, openai_splits)
            # print("----")
            self.evaluate_SBERT_vs_OpenAI_results(compare_file)

    #         TODO: hand these results back to Open Ai to generate an answer and compare
    #          qualitatively.  Does having more splits create a more accurate answer?  Which has
    #          better density of information- is the detail equal?  How do we score the density
    #          of technical information in the answer?

    def show_cossimilarityd_scores(self, SBERTList, OpenAIList):
        print("list of SBERT cosine similarity scores:")
        for sbert_item in SBERTList:
            print("sbert_item SplitID:", sbert_item.SplitID, "sbert_item CosineSimilarity:",
                  sbert_item.SplitCosineDistance)

        print("list of OpenAI cosine similarity scores:")
        for openai_item in OpenAIList:
            print("openai_item SplitID:", openai_item.SplitID, "openai_item CosineSimilarity:",
                  openai_item.SplitCosineDistance)

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

    def get_list_of_split_ids(self, SBERTList, OpenAIList)->list:
        split_id_list = []
        _seen_split_ids = set()
        for item in SBERTList:
            split_id = item.SplitID
            if split_id not in _seen_split_ids:
                _seen_split_ids.add(split_id)
                split_id_list.append(split_id)

        for item in OpenAIList:
            split_id = item.SplitID
            if split_id not in _seen_split_ids:
                _seen_split_ids.add(split_id)
                split_id_list.append(split_id)

        split_id_list.sort()
        return split_id_list


    def format_results_as_html_table(self, test_query, SBERTList, OpenAIList) -> Path:
        html_filename = Path(__file__).resolve().parent / f'results{self.html_document_number}.html'
        self.html_document_number += 1
        print(f'Writing results to {html_filename} with results for query: {test_query}')
        with open(html_filename, 'w', encoding='utf-8') as file:
            # write the HTML header with explicit newlines
            file.write("<!DOCTYPE html>\n<html>\n<body>\n")
            file.write(f'<h1>Results for query: {test_query}</h1>\n')
            file.write('<table border="1">\n')
            file.write(f'<tr><th width=50%>Algorithm One ({len(SBERTList)} splits)</th><th '
                       f'width=50%>Algorithm Two ('
                       f'{len(OpenAIList)} splits</th></tr>\n')

            SBERTList.sort(key=lambda item: getattr(item, "SplitID", float("inf")))
            OpenAIList.sort(key=lambda item: getattr(item, "SplitID", float("inf")))

            split_id_list = self.get_list_of_split_ids(SBERTList, OpenAIList)
            max_list_items = len(split_id_list)

            for i in range(max_list_items):
                sbert_text = ''
                openai_text = ''
                target_split_id = split_id_list[i]
                sbert_item = next((it for it in SBERTList if getattr(it, "SplitID", None) == target_split_id), None) #finds the first element in SBERTList whose SplitID equals target_split_id; if none is found, item becomes None.
                if sbert_item:
                    sbert_text = f'Distance={sbert_item.SplitCosineDistance}<br>{sbert_item.SplitContent}'

                openAI_item = next((it for it in OpenAIList if getattr(it, "SplitID", None) == target_split_id), None) #finds the first element in OpenAIList whose SplitID equals target_split_id; if none is found, item becomes None.
                if openAI_item:
                    openai_text = f'Distance={openAI_item.SplitCosineDistance}<br>{openAI_item.SplitContent}'

                # if OpenAIList[i].SplitID == target_split_id:
                #     openai_text = OpenAIList[i].SplitContent
                # sbert_item = SBERTList[i] if i < max_sbert_list_items else None
                # openai_item = OpenAIList[i] if i < max_openai_list_items else None
                # sbert_text = getattr(sbert_item, 'SplitContent', '') if sbert_item else ''
                # openai_text = getattr(openai_item, 'SplitContent', '') if openai_item else ''
                file.write(f'<tr><td>{sbert_text}</td><td>{openai_text}</td></tr>\n')

            file.write('</table>\n')
        file.close()
        return html_filename


    def evaluate_SBERT_vs_OpenAI_results(self, html_file):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        html_text = html_file.read_text()

        # Safety truncation to ensure we don't exceed Rate Limits (TPM) or Context Window
        # Limit to ~200k characters (approx 50k tokens) to remain safe.
        if len(html_text) > 200000:
            print(f"Warning: Input text too large ({len(html_text)} chars). Truncating to 200,000 chars.")
            html_text = html_text[:200000] + "\n...[TRUNCATED]..."

        prompt = ("Perform a deep analysis of the following html. It contains a table with "
                  "two columns of data."
                  "Each column contains text fragments from querying a large corpus of "
                  "information. "
                  "The two columns represent records generated by two different algorithms. "
                  "Ignore blank cells, and do not consider the raw number of rows returned as a "
                  "measure of relevance."
                  "Only base your evaluation on which algorithms created a set fragments with "
                  "the best "
                  "detailed and most relevant "
                  "information for use in creating a summary of the topic "
                  "contained in the first header of the HTML document."
                  "For your response, return the winner as either 'One' if the first column is "
                  "superior or 'Two' if the second column is superior.  Provide the winner "
                  "as "
                  "the first line with nothing else on it, followed by a detailed "
                  "explanation of your analysis\n"
                  "Here is the HTML document:\n"
                  "{html_text} \n\n")


        # Create a LangChain prompt template
        prompt_template = PromptTemplate(template=prompt,
                                         input_variables=["html_text"])

        # Create a LangChain LLMChain
        sequence = RunnableSequence(prompt_template | llm)

        # Get the response from the LLM
        response = sequence.invoke({"html_text": html_text})

        # Parse the JSON response
        response_data = response.content

        # Extract the winner and rationale
        winner = response_data.splitlines()[0].strip()
        rationale = "\n".join(response_data.splitlines()[1:])

        #Write response_data to html_file for further analysis
        with open(html_file, 'a', encoding='utf-8') as file:
            file.write(f'\n<br><h2>Evaluation Result:</h2>\n<pre>Winner:{winner}</pre>\n')
            file.write(f'\n<br>\n<pre>Rationale:\n{rationale}</pre>\n')
            file.write('</body>\n</html>\n')
        # Print the extracted elements
        print(f"Winner: {winner}")
        print(f"Rationale: {rationale}")
        return winner, rationale

if __name__ == '__main__':
    evaluator = SBERT_vs_OpenAI_evaluator()
    evaluator.setUp()
    evaluator.test_read_user_query_file()
