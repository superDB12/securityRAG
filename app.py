from datetime import datetime
from flask import Flask, request, render_template, redirect, url_for
from Experiments.generatorExperiments import Generator
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

app = Flask(__name__)
generator = Generator()

@app.route('/')
def index():
    return render_template('query.html')

@app.route('/query', methods=['GET', 'POST'])
def query():
    if request.method == 'POST':
        query_text = request.form['query']
        response = generator.generate_response(query_text)
        return render_template('results.html', query=query_text, result=response.content)

    return (render_template('query.html'))

@app.route('/results', methods=['GET', 'POST'])
def results():
    query = request.args.get('query')
    response = request.args.get('response')

    if request.method == 'POST':
        query = request.form['query']
        response = request.form['response']
        comments = request.form['comments']
        score = request.form['score']
        generator.request_response.add_request_and_response_log(request=query, date=datetime.now(),
                                                                response=response, comment=comments, score=int(score))
        return redirect(url_for('query'))

    return render_template('results.html', query=query, result=response)

if __name__ == '__main__':
    app.run(debug=True)