# Disinformation Detection RAG Chatbot

This project is a **Disinformation Detection RAG (Retrieve-Augmented Generation) Chatbot**. It leverages **Weaviate** as a vector database for document storage, **Google Gemini** for generative AI responses, and **Mesop UI** for the frontend interface. The chatbot is designed to detect and mitigate misinformation by analyzing articles, extracting factuality factors, and generating corrective responses.

## Features

- **Web Scraping with SerpApi**: Automatically scrape and validate claims from google search engine. Web scraped data from PolitiFact used for ICL.
- **Vector Database (Weaviate)**: Stores and retrieves article chunks with semantic embeddings for similarity-based search.
- **Fractal Chain of Thought (FCoT)**: Uses iterative prompting for advanced factuality analysis, incorporating all 12 Factuality Factors.
- **Google Gemini Integration**: Leverages Gemini for nuanced text generation and disinformation detection.
- **Mesop UI**: Provides an interactive chat interface to engage with the chatbot and get responses in real-time.

## Prerequisites

- Python 3.10+
- Weaviate Cloud Account
- Access to the following:
  - **Weaviate** instance (cloud only)
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
SERPAPI_KEY=your-serpapi-key
```

### Step 4: Run the Python Script
The Python script handles scraping, text chunking, vectorizing, and interaction with Weaviate and Google Gemini. To start the main application:
```bash
mesop src/backend/main.py
```

### Step 5: Use Mesop UI
Mesop UI provides the frontend for the chatbot. Once the Python backend is running, use Mesop's interface to interact with the system.

### Folder Structure

    .
    ├── src/                             # Main source code directory
    │   ├── backend/                     # Core backend logic
    │   │   ├── main.py                  # Main application file to run the project
    │   │   ├── Transform.py             # Script for transforming data
    │   │   ├── web_scrape.py            # Script for web scraping
    │   │   └── development/             # Temporary development and testing scripts
    │   │       ├── GenAI_chat.py
    │   │       ├── GenAI_Uploader.py
    │   │       └── llms.py
    │   ├── data/                        # Data directory
    │   │   ├── raw/                     # Raw data files
    │   │   │   ├── Factuality Factors and Implementation Strategies.pdf
    │   │   │   ├── small.tsv
    │   │   │   ├── test2_jsonl.jsonl
    │   │   │   ├── test2.tsv
    │   │   │   ├── train2_jsonl.jsonl
    │   │   │   ├── train2.tsv
    │   │   │   ├── val2_jsonl.jsonl
    │   │   │   └── val2.tsv
    │   │   └── processed/               # Processed and cleaned data
    │   │       ├── detailed_fact_checks.csv
    │   │       ├── gemini_system_prompt.txt
    │   │       ├── gemini_training_data.json
    │   │       └── pred_model.sav
    │   ├── predictive_model/            # Predictive model code
    │   │   ├── Authenticity_Modeling.ipynb
    │   │   ├── combined.ipynb           # Jupyter notebook for combine all models
    │   │   ├── factuality_factor_features.py 
    │   │   └── preditctive_model_content_stats.ipynb 
    ├── requirements.txt                 # Python dependencies
    ├── .env                             # Environment variables
    ├── .gitignore                       # Git ignore file
    └── README.md                        # Project overview and instructions



## Team
This project was developed by:

- **Yiheng Yuan**
- **Luran Zhang**
- **Jade Zhou**

### Mentor:

- **Dr. Ali Arsanjani**