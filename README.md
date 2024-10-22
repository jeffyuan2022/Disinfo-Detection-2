# Disinformation Detection RAG Chatbot

This project is a **Disinformation Detection RAG (Retrieve-Augmented Generation) Chatbot**. It leverages **Weaviate** as a vector database for document storage, **Google Gemini** for generative AI responses, and **Mesop UI** for the frontend interface.

## Features

- **Web Scraping**: Automatically scrape news articles from sources like CNN.
- **Vector Database (Weaviate)**: Stores and retrieves article chunks with semantic embeddings for similarity-based search.
- **Google Gemini Integration**: Uses the powerful Gemini model for text generation and disinformation detection.
- **Mesop UI**: Provides an interactive chat interface to engage with the chatbot and get responses in real-time.

## Prerequisites

- Python 3.7+
- Docker
- Access to the following:
  - **Weaviate** instance (local or cloud)
  - **Google Generative AI API** for Gemini
  - **Mesop UI** for chat interactions

### Libraries/Tools

The project uses the following Python libraries:

- `requests`
- `beautifulsoup4`
- `weaviate-client`
- `google-generativeai`
- `mesop-labs`
- `python-dotenv`
- `langchain`

To install the dependencies, you can use:

```bash
pip install -r requirements.txt
```

