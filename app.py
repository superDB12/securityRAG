from dataclasses import dataclass
from datetime import datetime
from flask import Flask, request, render_template, redirect, url_for
from Experiments.generatorExperiments import Generator
from logging.config import dictConfig
from database_access.requestAndResponseLogCRUD import RequestAndResponseLogCRUD
from database_access.session_factory import SessionFactory

from retriever.concept_extractor import ConceptExtractor

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

@dataclass
class RequestAndResponseData:
    Query: str
    QueryResponse: str
    QueryResponseScore: int
    QueryConcept: str
    QueryConceptResponse: str
    QueryConceptResponseScore: int
    Date: datetime

app = Flask(__name__)
generator = Generator()
request_and_response_log = RequestAndResponseLogCRUD(SessionFactory())


@app.route('/')
def index():
    return render_template('query.html')

@app.route('/query', methods=['GET', 'POST'])
def query():
    user_query = None
    if request.method == 'POST':
        user_query = request.form['query']

    # get the response from the user_query
    dc = RequestAndResponseData(
        Query="",
        QueryResponse="",
        QueryResponseScore=0,
        QueryConcept="",
        QueryConceptResponse="",
        QueryConceptResponseScore=0,
        Date=datetime.now()
    )

    dc.Query = user_query
    dc.QueryResponse = generator.generate_response(user_query)

    # get the response from the concept based on the user query
    my_concept_extractor = ConceptExtractor()
    dc.QueryConcept = my_concept_extractor.extract_concept(user_query)
    dc.QueryConceptResponse = generator.generate_response_from_query_and_concept(user_query,
                                                                                 dc.QueryConcept)

    return render_template('results.html', query=user_query, result=dc)

    return (render_template('query.html'))

@app.route('/results', methods=['GET', 'POST'])
def results():
    if request.method == 'POST':
        user_query = request.form.get('user_query')
        query_response = request.form.get('query_response')
        query_concept = request.form.get('query_concept')
        query_concept_response = request.form.get('query_concept_response')
        query_score = request.form.get('query_score', 0, type=int)
        query_concept_score = request.form.get('concept_score', 0, type=int)
        query_score_comments = request.form.get('query_score_comments', '')
        concept_score_comments = request.form.get('concept_score_comments', '')

        request_and_response_log.add_query_and_response_log(
            query=user_query,
            query_response=query_response,
            query_response_score=query_score,
            query_response_comments= query_score_comments,
            query_concept=query_concept,
            query_concept_response=query_concept_response,
            concept_score=query_concept_score,
            concept_score_comments= concept_score_comments,
            date=datetime.now()
        )
        return redirect(url_for('index'))

    return render_template('results.html', query=query, result=response)

if __name__ == '__main__':
    app.run(debug=True)