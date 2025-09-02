# Chatbot with RAG Pipeline

This project is a **Retrieval-Augmented Generation (RAG) based chatbot** that answers user queries by combining **vector search** over research paper content with **OpenAI GPT models**. It uses **PostgreSQL with pgvector** to store embeddings and provides context-aware, conversational responses.


## Features

* Store and search research paper content using vector embeddings.
* Retrieve the most relevant chunks of text for a given question.
* Generate responses with GPT even when no relevant document is found.
* Clear and modular pipeline for embedding, retrieval, and generation.
* Simplified integration: `sentence-transformers` for embeddings, OpenAI for LLM.


## Project Structure

chatbot-with-rag-pipeline/
├── app.py                  # Main chatbot application
├── pipeline/               # Embedding, retrieval, and utility modules
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── .gitignore              # Ignore sensitive files and caches


## Setup Instructions

### 1. Clone Repository

git clone (https://github.com/AhmadSohaib78/Chatbot-with-RAG-Pipeline.git)
cd chatbot-with-rag-pipeline

### 2. Install Dependencies

pip install -r requirements.txt

### 3. Configure Environment Variables

In the code, replace the placeholder with your own keys:

client = OpenAI(api_key="YOUR_API_KEY_HERE")

For PostgreSQL connection, update your connection string:

DATABASE_URL = "postgresql://username:password@localhost:5432/dbname"

### 4. Run Application

python app.py

## How It Works

1. **Embedding**
   Text chunks from PDFs are converted into vector embeddings using `sentence-transformers` models (e.g., `all-MiniLM-L6-v2`).

2. **Vector Store**
   Embeddings are stored in PostgreSQL with the `pgvector` extension, enabling efficient similarity search.

3. **Retrieval**
   For a user query, the system retrieves the most relevant embeddings from the database.

4. **LLM Response**
   The retrieved context, along with the question, is sent to an OpenAI GPT model.

   * If relevant matches exist → GPT uses them to generate a contextual response.
   * If no matches exist → GPT still answers using only the question.

## Example Usage

**Query:**
"What is the main contribution of the paper?"

**Response:**
GPT searches the embeddings for related chunks. If found, it answers using the context. If none are found, it generates a helpful response directly.

## Requirements

* Python 3.9+
* PostgreSQL with pgvector extension
* Dependencies listed in `requirements.txt`

Example requirements.txt:

openai>=1.0.0
psycopg2
sentence-transformers
langchain
python-dotenv

## Future Improvements

* Add support for multiple embedding models.
* Provide a web interface for user interaction.
* Improve context ranking using hybrid search (BM25 + embeddings).
* Enable fine-tuned GPT prompts for specific domains.
