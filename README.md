# Disinformation Detection RAG Chatbot

This project is a **Disinformation Detection RAG (Retrieve-Augmented Generation) Chatbot**. It leverages **Weaviate** as a vector database for document storage, **Google Gemini** for generative AI responses, and **Mesop UI** for the frontend interface.

## Features

- **Web Scraping**: Automatically scrape news articles from sources like CNN.
- **Vector Database (Weaviate)**: Stores and retrieves article chunks with semantic embeddings for similarity-based search.
- **Google Gemini Integration**: Uses the powerful Gemini model for text generation and disinformation detection.
- **Mesop UI**: Provides an interactive chat interface to engage with the chatbot and get responses in real-time.

## Prerequisites

- Python 3.10+
- Weaviate Cloud Account or Docker
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
GOOGLE_GENAI_API_KEY=your-google-genai-api-key
WEAVIATE_API_KEY=your-weaviate-api-key
WEAVIATE_CLUSTER=your-cloud-instance-URL
```

### Step 3: Run Weaviate with Docker (Optional)
If you're running Weaviate locally, please follow [**this link**](https://weaviate.io/developers/weaviate/installation/docker-compose) to setup. Then replace 
```
weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=wcd_url,
    auth_credentials=weaviate.classes.init.Auth.api_key(wcd_api_key)
)
``` 
with 
```
client = weaviate.connect_to_local()
``` 
in `backend/Combined_model.py` file.

### Step 4: Run the Python Script
The Python script handles scraping, text chunking, vectorizing, and interaction with Weaviate and Google Gemini. To start the script:
```bash
mesop backend/Combined_model.py
```

### Step 5: Use Mesop UI
Mesop UI provides the frontend for the chatbot. Once the Python backend is running, use Mesop's interface to interact with the system.

## Team
This project was developed as part of a team effort. The following individuals contributed to the development of this disinformation detection chatbot:

- **Yiheng Yuan**
- **Luran Zhang**
- **Jade Zhou**

### Mentor:

- **Dr. Ali Arsanjani**