import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import json
import html
import weaviate
# from weaviate.classes.init import Auth (Weaviate Cloud Server)
import weaviate.classes.config as wc
# from weaviate.util import generate_uuid5
from typing import List
import google.generativeai as genai
import mesop as me
import mesop.labs as mel
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GOOGLE_GENAI_API_KEY = os.getenv("GOOGLE_GENAI_API_KEY")

article_url = "https://www.cnn.com/2024/10/15/politics/harris-oil-companies-emissions-kfile/index.html"
response = requests.get(article_url)

if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')
    json_ld_tag = soup.find('script', {'type': 'application/ld+json'})

    if json_ld_tag:
        json_content = json.loads(json_ld_tag.string)
        for item in json_content:
            if item.get('@type') == 'NewsArticle':
                headline = item.get('headline', 'No headline found')
                description = item.get('description', 'No description found')
                author = item.get('author', [{'name': 'No author found'}])[0]['name']
                date_modified = item.get('dateModified', 'No date found')
                article_body = item.get('articleBody', 'No article body found')

                clean_article_body = html.unescape(article_body)
                clean_article_body = clean_article_body.replace('\xa0', ' ')

                print(f"Headline: {headline}")
                print(f"Description: {description}")
                print(f"Author: {author}")
                print(f"Date Modified: {date_modified}")
                print(f"Cleaned Article Body: {clean_article_body[:200]}...")  # Printing the first 200 characters
                break
    else:
        print("No JSON-LD data found on the page.")
else:
    print(f"Failed to retrieve article, status code: {response.status_code}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 20)
text_chunks = text_splitter.split_text(clean_article_body)

client = weaviate.connect_to_local()

assert client.is_live()

# testing

client.collections.delete_all()
client.collections.create(
        "CNNArticle",
        properties=[
            wc.Property(name="content", data_type=wc.DataType.TEXT),
            wc.Property(name="chunk_id", data_type=wc.DataType.INT),
            wc.Property(name="source", data_type=wc.DataType.TEXT)
        ],
        vectorizer_config=wc.Configure.Vectorizer.text2vec_google(
            api_key=GOOGLE_GENAI_API_KEY,
            project_id=PROJECT_ID
        ),
        generative_config=wc.Configure.Generative.google(
            api_key=GOOGLE_GENAI_API_KEY,
            project_id=PROJECT_ID
        )
    )

def clean_text_chunk(chunk: str) -> str:
        """Clean and validate text chunk."""
        if not isinstance(chunk, str):
            return ""
        # Remove any null bytes
        chunk = chunk.replace('\x00', '')
        # Limit chunk size if needed
        max_length = 25000
        if len(chunk) > max_length:
            chunk = chunk[:max_length]
        return chunk.strip()

collection = client.collections.get("CNNArticle")
    
def process_and_import_text(collection, text_chunks: List[str], source: str = "article", chunk_size: int = 10):
    """Import text chunks to Weaviate collection with proper formatting and error handling."""
    total_chunks = len(text_chunks)
    successful_imports = 0
    failed_imports = 0
    
    try:
        # Process chunks in smaller batches
        for batch_start in range(0, total_chunks, chunk_size):
            batch_end = min(batch_start + chunk_size, total_chunks)
            current_batch = text_chunks[batch_start:batch_end]
            
            with collection.batch.dynamic() as batch:
                for idx, chunk in enumerate(current_batch, start=batch_start):
                    # Clean and validate the chunk
                    clean_chunk = clean_text_chunk(chunk)
                    if not clean_chunk:
                        print(f"Skipping empty or invalid chunk at index {idx}")
                        continue
                    
                    try:
                        weaviate_obj = {
                            "content": clean_chunk,
                            "chunk_id": idx,
                            "source": source
                        }
                        batch.add_object(properties=weaviate_obj)

                    except Exception as e:
                        print(f"Error adding object at index {idx}: {str(e)}")
                        failed_imports += 1
                        continue
                    
                    successful_imports += 1
                batch.flush()
                print(f"Processed batch ending at index {batch_end - 1}")

            print(f"Processed batch {batch_start//chunk_size + 1}/{(total_chunks + chunk_size - 1)//chunk_size}")
            
        print(f"Import complete. Successful: {successful_imports}, Failed: {failed_imports}")
        
        if hasattr(batch, 'failed_objects') and batch.failed_objects:
            print("Failed objects details:")
            for obj in batch.failed_objects:
                print(f"Failed object: {obj}")
                
    except Exception as e:
        print(f"Batch import error: {str(e)}")
        raise

process_and_import_text(collection, text_chunks, chunk_size=10)

CNN = client.collections.get("CNNArticle")

# # Perform query
# response = CNN.generate.near_text(
#     query="Fake News",
#     limit=5,
#     grouped_task="What do these chunk have in common?",
#     # grouped_properties=["title", "overview"]  # Optional parameter; for reducing prompt length
# )

# # Inspect the response
# for o in response.objects:
#     print(o.properties["title"])  # Print the title
# print(response.generated)  # Print the generated text (the commonalities between them)

genai.configure(api_key=GOOGLE_GENAI_API_KEY)

generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash-002",
  generation_config=generation_config,
)

chat_session = model.start_chat(history=[])

def get_generated_response(prompt: str, collection) -> str:
    try:
        response = collection.generate.near_text(
            query=prompt,
            grouped_task="Provide a detailed response based on the query and relevant content.",
            limit=2  # retrieve 2 relevant chunks
        )

        generated_text = response.generated
        return generated_text
    except Exception as e:
        print(f"Error generating response from Weaviate: {str(e)}")
        return "Sorry, there was an issue generating the response."

@me.page(
    security_policy=me.SecurityPolicy(
        allowed_iframe_parents=["https:google.github.io"]
    ),
    path="/chat",
    title= "Mesop Demo",
)

def page():
    mel.chat(transform)

def transform(prompt: str, history: list[mel.ChatMessage]) -> str:
    generated_response = get_generated_response(prompt, collection)
    # chat_history = "\n".join(message.content for message in history)
    # full_input = f"{chat_history}\n{generated_response}"
    # response = chat_session.send_message(full_input)
    return generated_response