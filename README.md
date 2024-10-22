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

## Setup Instructions

### Step 1: Clone the Repository
```bash
git clone https://github.com/jeffyuan2022/Disinfo-Detection.git
cd Disinfo-Detection
```

### Step 2: Set Up Environment Variables
Create a `.env` file in the root directory to store your sensitive information (API keys, project IDs, etc.).
```bash
touch .env
```
Populate the `.env` file with the following variables:
```bash
# .env
WEAVIATE_URL=http://localhost:8080  # or your cloud instance URL
GCP_PROJECT_ID=your-gcp-project-id
GOOGLE_GENAI_API_KEY=your-google-genai-api-key
```

### Step 3: Run Weaviate with Docker
If you're running Weaviate locally, use the following command to start Weaviate:
```bash
docker run -d \
  -p 8080:8080 \
  -e QUERY_DEFAULTS_LIMIT=20 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
  -v $(pwd)/weaviate-data:/var/lib/weaviate \
  semitechnologies/weaviate:latest
```

### Step 4: Run the Python Script
The Python script handles scraping, text chunking, vectorizing, and interaction with Weaviate and Google Gemini. To start the script:
```bash
mesop backend/app.py
```

### Step 5: Use Mesop UI
Mesop UI provides the frontend for the chatbot. Once the Python backend is running, use Mesop's interface to interact with the system.

## Team
This project was developed as part of a team effort. The following individuals contributed to the development of this disinformation detection chatbot:

- **Yiheng Yuan**
- **Luren Zhang**
- **Jade Zhou**

### Mentor:

- **Dr. Ali Arsanjani**