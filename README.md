# SecurityRAG

SecurityRAG is a Retrieval-Augmented Generation (RAG) application designed to search and retrieve information from a repository of security-related documents. It leverages the power of large language models (LLMs) to provide accurate and contextually relevant answers to user queries.

## Key Features

*   **RAG-based Architecture:** Utilizes a RAG architecture to combine the strengths of a retriever and a generator for enhanced information retrieval.
*   **Vector-based Search:** Employs vector embeddings to perform efficient similarity searches on a corpus of security documents.
*   **Concept Extraction:** Extracts the core concepts from user queries to improve the accuracy of search results.
*   **Database Integration:** Uses a PostgreSQL database with the `pgvector` extension to store and manage vector embeddings.
*   **Flask-based Web Interface:** Provides a simple and intuitive web interface for users to interact with the a pplication.

## Technologies Used

*   **Python:** The primary programming language for the application.
*   **Flask:** A lightweight web framework for building the user interface.
*   **LangChain:** A library for building applications with LLMs.
*   **SQLAlchemy:** A SQL toolkit and Object-Relational Mapper (ORM) for Python.
*   **pgvector:** A PostgreSQL extension for vector similarity search.
*   **Sentence-Transformers:** A Python framework for state-of-the-art sentence, text, and image embeddings.
*   **OpenAI:** Provides the large language models used for generation and embedding.

## Architecture

The application is divided into several components, each with a specific responsibility:

*   **`app.py`:** The main entry point of the application. It handles the web server, routing, and user interface.
*   **`generator`:** This component is responsible for generating responses to user queries. It uses a large language model to generate text based on the retrieved information.
*   **`retriever`:** The retriever is responsible for finding relevant information from the document repository. It uses a vector-based search to find the most similar documents to the user's query.
*   **`database_access`:** This component handles all interactions with the database. It provides a set of CRUD (Create, Read, Update, Delete) operations for managing documents, splits, and embeddings.

## Getting Started

Follow these instructions to set up and run the application on your local machine.

### Prerequisites

*   Python 3.8 or higher
*   A running PostgreSQL instance with the `pgvector` extension enabled
*   Access to the Google Cloud Secret Manager API
*   An OpenAI API key

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/securityRAG.git
    cd securityRAG
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**

    Create a `.env` file in the root of the project and add the following variables:

    ```
    PROJECT_ID=<Your Google Cloud project ID>
    REGION=<Your Google Cloud region>
    INSTANCE_NAME=<Your Cloud SQL instance name>
    MAX_SPLITS=<Maximum number of splits to retrieve>
    DIST_THRESHOLD=<Distance threshold for similarity search>
    ```

5.  **Set up your secrets in Google Cloud Secret Manager:**

    You will need to create secrets for the following:

    *   `DATABASE_NAME`
    *   `DB_USERNAME`
    *   `DB_PASSWORD`
    *   `OPENAI_API_KEY`

### Running the Application

Once you have completed the setup, you can run the application with the following command:

```bash
flask run
```

The application will be available at `http://127.0.0.1:5000`.

## Database

The application uses a PostgreSQL database to store the following information:

*   **`documents`:** This table stores the original documents that are added to the repository.
*   **`split_documents`:** This table stores the splits of the original documents. Each document is divided into smaller chunks, or splits, to facilitate vector-based search.
*   **`embeddings`:** This table stores the vector embeddings for each split. The embeddings are generated using a pre-trained language model and are used to perform similarity searches.
*   **`request_and_response_log`:** This table logs all user queries and the corresponding responses from the application. This information can be used for analysis and debugging.
