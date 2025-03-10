import PyPDF2
import mesop as me
import google.generativeai as genai
from typing import List
import os
from io import BytesIO
from dotenv import load_dotenv
import pickle
import numpy as np
import pandas as pd
import json
from dataclasses import field
import nltk
import article_extractor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from textblob import TextBlob
import spacy
from collections import Counter
import re
import time
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
import weaviate
from weaviate.classes.config import Property, DataType
import weaviate.classes.config as wvc
from weaviate.util import generate_uuid5
import requests

# Load your predictive model
def load_predictive_model():  
    filename = 'src/data/processed/pred_model.sav'
    model = pickle.load(open(filename, 'rb'))
    return model

predictive_model = load_predictive_model()

# Load API key from environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_GENAI_API_KEY"))

# Initialize the generative model
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

@me.stateclass
class State:
    file: me.UploadedFile = None
    url: str = ""
    selected_input: str = ""  # "pdf" or "url"
    article_title: str = ""
    article_text: str = ""
    error_message: str = ""
    loading: bool = False
    Final_Score: float = 0.0
    analysis_results: pd.DataFrame = field(default_factory=pd.DataFrame)

def load(e: me.LoadEvent):
    me.set_theme_mode("system")

@me.page(
  on_load=load,
  security_policy=me.SecurityPolicy(
    allowed_iframe_parents=["https://google.github.io"]
  ),
  path="/",
)
def app():
    state = me.state(State)

    with me.box(style=me.Style(padding=me.Padding.all(15))):
        
        # PDF Uploader Section
        me.text("Upload a PDF file for analysis", type="headline-4")
        me.uploader(
            label="Upload PDF File",
            accepted_file_types=["application/pdf"],
            on_upload=handle_upload,
            type="flat",
            color="primary",
            style=me.Style(font_weight="bold"),
            disabled=state.selected_input == "url"  # Disable if user selected URL
        )

        if state.file and state.file.size:
            with me.box(style=me.Style(margin=me.Margin.all(10))):
                me.text(f"File name: {state.file.name}")
                # me.text(f"File size: {state.file.size} bytes")
                # me.text(f"File type: {state.file.mime_type}")

        # Divider
        me.text("Or", type="headline-5")

        # URL Input Section
        me.text("Enter an article URL for analysis", type="headline-4")
        me.input(
            label="Enter Article URL",
            value=state.url,
            on_input=lambda event: select_url(event.value),
            style=me.Style(width="100%"),
            disabled=state.selected_input == "pdf"  # Disable if user uploaded a PDF
        )

        # Buttons Section
        with me.box(style=me.Style(margin=me.Margin.all(15))):
            me.button(
                on_click=start_analysis,
                label="Start Analysis",
                disabled=not (state.file or state.url),  # Enable only if one input is provided
                style=me.Style(
                    background="#007bff", 
                    color="white", 
                    font_size=16
                ),
            )
            # New Reset Button
            me.button(
                on_click=reset_all,
                label="Reset",
                style=me.Style(
                    background="#dc3545",
                    color="white",
                    font_size=16,
                    margin=me.Margin.all(10)
                ),
            )

        if state.loading:
            me.progress_spinner(style=me.Style(margin=me.Margin.all(24)))

        if state.error_message:
            with me.box(style=me.Style(background="#f8d7da", padding=me.Padding.all(12))):
                me.text(text=f"Error: {state.error_message}", type="body-1")

        if state.article_title:
            me.text(text=state.article_title, type="headline-3")

        # if state.article_text:
        #     me.textarea(value=state.article_text, readonly=True, rows=20)

        # If analysis results exist, display them
        if state.analysis_results is not None and not state.analysis_results.empty:
            with me.box(style=me.Style(padding=me.Padding.all(10))):
                me.table(
                    state.analysis_results,
                    header=me.TableHeader(sticky=True),
                    columns={col: me.TableColumn(sticky=True) for col in state.analysis_results.columns},
                )
            me.text("Final Veracity Score (1 to 6): " + str(state.Final_Score))

def reset_all(event=None):
    """Resets the application state and clears data from Weaviate."""
    state = me.state(State)
    # Reset state variables
    state.file = None
    state.url = ""
    state.selected_input = ""
    state.article_title = ""
    state.article_text = ""
    state.error_message = ""
    state.loading = False
    state.Final_Score = 0.0
    state.analysis_results = pd.DataFrame()

    # Reset Weaviate data for ArticleChunk
    try:
        # Check if the collection exists and delete it
        collections = weaviate_client.collections.list_all()
        if "ArticleChunk" in collections:
            weaviate_client.collections.delete("ArticleChunk")
            print("ArticleChunk collection deleted.")
        # Recreate the collection
        create_article_chunk_schema(weaviate_client)
        print("ArticleChunk collection reinitialized successfully.")
    except Exception as e:
        print(f"Error resetting Weaviate data: {e}")

def handle_upload(event: me.UploadEvent):
    """Handles PDF upload and ensures URL input is cleared."""
    state = me.state(State)
    
    # If user uploads a PDF, clear URL input
    state.url = ""
    state.selected_input = "pdf"
    
    state.file = event.file

def select_url(url: str):
    """Handles URL input and ensures PDF upload is cleared."""
    state = me.state(State)
    
    # If user enters a URL, clear uploaded file
    state.file = None
    state.selected_input = "url"
    
    state.url = url

def start_analysis(event=None):  # Accept event argument to avoid the error
    """Starts analysis based on the selected input type."""
    state = me.state(State)

    if state.selected_input == "pdf" and state.file:
        results = process_pdf_and_analyze(state.file)
    elif state.selected_input == "url" and state.url:
        results = extract_and_analyze_url(state.url)
    else:
        state.error_message = "No valid input selected for analysis."
        return

    update_analysis_results(results)

def extract_and_analyze_url(url: str):
    """Extracts article from URL and analyzes it."""
    state = me.state(State)
    state.loading = True
    state.article_title = ""
    state.article_text = ""
    state.error_message = ""

    result = article_extractor.extract_article(url)

    state.loading = False
    if result["error"]:
        state.error_message = result["error"]
        return None
    else:
        state.article_title = result["title"]
        state.article_text = result["text"]
        return process_text_and_analyze(state.article_text)

def update_analysis_results(results):
    """Stores the analysis results in the state."""
    state = me.state(State)
    if results:
        state.analysis_results = pd.DataFrame([
            ["Biases", results["Overall Biases Score"], results["Biases Consideration"],
             "Google Gemini 1.5 pro with function calling and FCoT prompting"],
            ["Context Veracity", results["Overall Context Veracity Score"], results["Context Veracity Consideration"],
             "Google Gemini 1.5 pro with function calling and FCoT prompting"],
            ["Information Utility", results["Overall Information Utility Score"], results["Information Utility Consideration"],
             "Google Gemini 1.5 pro with function calling and FCoT prompting"],
            ["Content Statistic", results["Overall Content Statistic Score"], "N/A",
             "Predictive models trained from Liar Plus Dataset"],
            ["Authenticity", results["Overall Authenticity Score"], "N/A",
             "Predictive models trained from Liar Plus Dataset"],
            ["Linguistic Based", results["Overall Linguistic Based Score"], "N/A",
             "Predictive models trained from Liar Plus Dataset"],
        ], columns=["Factuality Factors", "Veracity Score (1 to 6)", "Consideration", "Citation"])
        state.Final_Score = results["Final Score"]


# Function to process PDF files
def process_pdf_and_analyze(file: me.UploadedFile):
    pdf_text = extract_text_from_pdf(file)
    return process_text_and_analyze(pdf_text)

# Function to process extracted text from URL
def process_text_and_analyze(text: str):
    chunks = chunk_text(text)
    store_chunks_in_weaviate(weaviate_client, chunks)
    stored_chunks = get_chunks_from_weaviate(weaviate_client)

    if not stored_chunks:
        return {"error": "No valid chunks found for analysis."}

    # Initialize aggregated results for generative and predictive factors
    total_biases_score = 0.0
    total_context_veracity_score = 0.0
    total_info_utility_score = 0.0
    total_ai_score = 0.0
    total_predictive_score = 0.0

    # New totals for additional predictive factors
    total_content_stat_score = 0.0
    total_authenticity_score = 0.0
    total_linguistic_score = 0.0

    # Collect explanations for summary (only for generative factors)
    biases_explanations = []
    context_explanations = []
    utility_explanations = []

    total_chunks = len(stored_chunks)

    for i, chunk in enumerate(stored_chunks):
        # Step 5.1: Get AI model's structured JSON output (generative analysis)
        ai_analysis = analyze_chunk_with_gemini(chunk)
        
        if not isinstance(ai_analysis, dict):
            print(f"Error processing chunk {i+1}: AI analysis output is invalid.")
            continue  # Skip invalid results

        # Step 5.2: Get predictive model's score (original predictive model)
        predictive_score = get_predictive_model_score(chunk)

        # Step 5.3: Extract AI score from analysis
        ai_score = ai_analysis.get("Overall_Score", 0.0)

        # Step 5.4: Aggregate Generative Scores
        total_biases_score += ai_analysis["Biases_Factuality_Factor"]["Overall_Score"]
        total_context_veracity_score += ai_analysis["Context_Veracity_Factor"]["Overall_Score"]
        total_info_utility_score += ai_analysis["Information_Utility_Factor"]["Overall_Score"]
        total_ai_score += ai_score
        total_predictive_score += predictive_score

        # Step 5.5: Aggregate Predictive Factor Scores from custom functions
        # For articles from URL/PDF, we pass default values for job title and justification.
        content_stat_score = compute_factuality_score(chunk)
        authenticity_score = compute_authenticity_score(chunk, "Unknown", "None", politifact_data)
        linguistic_score = compute_linguistic_persuasion_score(chunk)

        total_content_stat_score += content_stat_score
        total_authenticity_score += authenticity_score
        total_linguistic_score += linguistic_score

        # Step 5.6: Collect Explanations for Generative Analysis Summary
        biases_explanations.append(". ".join([
            ai_analysis["Biases_Factuality_Factor"]["Language_Analysis"]["Explanation"],
            ai_analysis["Biases_Factuality_Factor"]["Tonal_Analysis"]["Explanation"],
            ai_analysis["Biases_Factuality_Factor"]["Balanced_Perspective_Checks"]["Explanation"]
        ]))

        context_explanations.append(". ".join([
            ai_analysis["Context_Veracity_Factor"]["Consistency_Checks"]["Explanation"],
            ai_analysis["Context_Veracity_Factor"]["Contextual_Shift_Detection"]["Explanation"],
            ai_analysis["Context_Veracity_Factor"]["Setting_Based_Validation"]["Explanation"]
        ]))

        utility_explanations.append(". ".join([
            ai_analysis["Information_Utility_Factor"]["Content_Value"]["Explanation"],
            ai_analysis["Information_Utility_Factor"]["Cost_Analysis"]["Explanation"],
            ai_analysis["Information_Utility_Factor"]["Reader_Value"]["Explanation"]
        ]))

    # Step 6: Compute Average Scores for each factor
    avg_biases_score = total_biases_score / total_chunks
    avg_context_veracity_score = total_context_veracity_score / total_chunks
    avg_info_utility_score = total_info_utility_score / total_chunks
    avg_ai_score = total_ai_score / total_chunks
    avg_predictive_score = total_predictive_score / total_chunks
    print("avg_predictive_score", avg_predictive_score)
    avg_predictive_score_adj = round(min(max((avg_predictive_score * 5) + 1, 1), 6), 1)
    print("avg_predictive_score_adj", avg_predictive_score_adj)
    avg_content_stat_score = total_content_stat_score / total_chunks
    avg_authenticity_score = total_authenticity_score / total_chunks
    avg_linguistic_score = total_linguistic_score / total_chunks

    # Final score: average of the six factors (adjust weights as desired)
    final_score = round(((avg_biases_score + avg_context_veracity_score + avg_info_utility_score + 
                   avg_predictive_score_adj) / 4), 0)

    # Step 7: Create Final Report as a Single Output
    final_report = {
        "Overall Biases Score": avg_biases_score,
        "Biases Consideration": " ".join(biases_explanations),
        "Overall Context Veracity Score": avg_context_veracity_score,
        "Context Veracity Consideration": " ".join(context_explanations),
        "Overall Information Utility Score": avg_info_utility_score,
        "Information Utility Consideration": " ".join(utility_explanations),
        "Overall Content Statistic Score": avg_content_stat_score,
        "Content Statistic Consideration": "N/A",
        "Overall Authenticity Score": avg_authenticity_score,
        "Authenticity Consideration": "N/A",
        "Overall Linguistic Based Score": avg_linguistic_score,
        "Linguistic Based Consideration": "N/A",
        "Final Score": final_score
    }

    return final_report  # Return a single aggregated analysis

def extract_text_from_pdf(file: me.UploadedFile) -> str:
    pdf_stream = BytesIO(file.getvalue())
    reader = PyPDF2.PdfReader(pdf_stream)
    return " ".join([page.extract_text() for page in reader.pages])

def chunk_text(text: str, chunk_size=2000) -> List[str]:
    sentences = sent_tokenize(text)
    return [" ".join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]

## Testing for Weaviate Database
wcd_url = os.getenv("WEAVIATE_CLUSTER")
wcd_api_key = os.getenv("WEAVIATE_API_KEY")

# Initialize Weaviate Client
weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=wcd_url,
    auth_credentials=weaviate.classes.init.Auth.api_key(wcd_api_key),
    skip_init_checks=True
)

# Define the collection schema for ArticleChunk
def create_article_chunk_schema(client):
    # Create the ArticleChunk collection with text vectorization and generative configuration
    client.collections.create(
        name="ArticleChunk",
        # vectorizer_config=wvc.Configure.Vectorizer.text2vec_openai(),
        # generative_config=wvc.Configure.Generative.openai(),
        properties=[
            Property(name="text", data_type=DataType.TEXT),
        ]
    )

# Define the collection schema for LiarPlusClaim
def create_liar_plus_schema(client):
    client.collections.create(
        name="LiarPlus",
        vectorizer_config=wvc.Configure.Vectorizer.text2vec_openai(),
        generative_config=wvc.Configure.Generative.openai(),
        properties=[
            Property(name="claim", data_type=DataType.TEXT),
            Property(name="label", data_type=DataType.TEXT),
            Property(name="topics", data_type=DataType.TEXT),
            Property(name="originator", data_type=DataType.TEXT),
            Property(name="title", data_type=DataType.TEXT),
            Property(name="party", data_type=DataType.TEXT),
            Property(name="justification", data_type=DataType.TEXT),
        ],
    )

# Delete old schema and create new schema
# weaviate_client.collections.delete("ArticleChunk")
# weaviate_client.collections.delete("LiarPlus")
# create_article_chunk_schema(weaviate_client)
# create_liar_plus_schema(weaviate_client)

"""
Check if the 'ArticleChunk' cluster exists in Weaviate Cloud.
If it exists, delete it and recreate it.
If it does not exist, create it.
"""
try:
    # Check if the cluster exists
    response = weaviate_client.collections.list_all()
    if "ArticleChunk" in response:
        weaviate_client.collections.delete("ArticleChunk")
        print("Old Cluster deleted.")

    # Create the cluster
    print("Creating the 'ArticleChunk' cluster...")
    create_article_chunk_schema(weaviate_client)
    print("Cluster 'ArticleChunk' initialized successfully.")

except Exception as e:
    print(f"An error occurred while initializing the 'ArticleChunk' cluster: {e}")

# Store chunked text in Weaviate using the dynamic batch context
def store_chunks_in_weaviate(client, chunks: List[str]):
    collection = client.collections.get("ArticleChunk")
    
    with collection.batch.dynamic() as batch:
        for chunk in chunks:
            # Generate a unique UUID for each chunk
            obj_uuid = generate_uuid5({"text": chunk})
            batch.add_object(
                properties={"text": chunk},
                uuid=obj_uuid
            )

# Store liar plus in Weaviate using the dynamic batch context
def store_liar_plus_in_weaviate(client, filepath):
    collection = client.collections.get("LiarPlus")
    
    with collection.batch.dynamic() as batch:
        with open(filepath, 'r') as f:
            for line in f:
                claim_data = json.loads(line.strip())

                batch.add_object(
                    properties={
                        "claim": claim_data["claim"],
                        "label": claim_data["label"],
                        "topics": claim_data["topics"],
                        "originator": claim_data["originator"],
                        "title": claim_data["title"],
                        "party": claim_data["party"],
                        "justification": claim_data["justification"]                        
                    },
                    uuid=generate_uuid5(claim_data["id"])
                )
                failed_objs_a = collection.batch.failed_objects  # Get failed objects
                if failed_objs_a:
                    print(f"Number of failed objects in the first batch: {len(failed_objs_a)}")
                    for i, failed_obj in enumerate(failed_objs_a, 1):
                        print(f"Failed object {i}:")
                        print(f"Error message: {failed_obj.message}")
                else:
                    print("All objects were successfully added.")

# store_liar_plus_in_weaviate(weaviate_client, "./data/train2_jsonl.jsonl")

def retrieve_and_analyze_claim(chunk_text):
    response = weaviate_client.query.get("LiarPlus", ["claim", "label", "justification", "topics", "originator", "title", "party"]) \
        .with_near_text({"concepts": [chunk_text]}) \
        .with_limit(3) \
        .do()
    
    # Process the results
    results = []
    for result in response["data"]["Get"]["LiarPlus"]:
        claim_text = result["claim"]
        label = result["label"]
        justification = result["justification"]
        confidence = 1 if label == "true" else 0  # Simplified confidence based on label
        
        results.append({
            "claim": claim_text,
            "label": label,
            "confidence": confidence,
            "justification": justification
        })
    
    return results

# Example Usage
# chunk_text = "Sample text from a chunk in PDF"
# similar_claims = retrieve_and_analyze_claim(chunk_text)
# print(similar_claims)

# Retrieve chunked text from Weaviate
def get_chunks_from_weaviate(client) -> List[str]:
    # Get the collection for querying
    article_chunk_collection = client.collections.get("ArticleChunk")
    
    # Fetch objects with a specified limit and return only the "text" property
    response = article_chunk_collection.query.fetch_objects(
        limit=100,  # Set limit as needed
        return_properties=["text"]
    )
    
    # Extract the "text" property from each returned object
    chunks = [obj.properties["text"] for obj in response.objects if "text" in obj.properties]
    return chunks
## End of testing

def context_cross_check(claims: list, max_results: int = 3) -> dict:
    """
    Cross-check claims using a single SerpApi call, limiting results for efficiency.
    
    Parameters:
        claims (list): List of claims to verify.
        max_results (int): Maximum number of results to retrieve for each claim.
    
    Returns:
        dict: Results with claims as keys and verification status as values.
    """
    serpapi_api_key = os.getenv("SERPAPI_API_KEY")
    search_url = "https://serpapi.com/search"  # SerpApi endpoint

    # Combine all claims into a single query
    combined_query = " OR ".join(claims) # Use "OR" to search for multiple claims in one query
    
    try:
        # Perform a single API request for the combined query
        params = {
            "q": combined_query,
            "engine": "google",
            "api_key": serpapi_api_key,
            "num": max_results  # Limit results to top `max_results`
        }
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        search_results = response.json()
        # Parse the search results
        results = {}
        if "organic_results" in search_results:
            for claim in claims:
                # Check if any of the top results mention the claim
                matched_results = [
                    result for result in search_results["organic_results"] 
                    if claim.lower() in result.get("title", "").lower() or 
                       claim.lower() in result.get("snippet", "").lower()
                ]
                results[claim] = "Verified" if matched_results else "Not Verified"
        else:
            # No results found
            results = {claim: "No Data" for claim in claims}
        
    except Exception as e:
        # Handle errors gracefully
        results = {claim: f"Error: {str(e)}" for claim in claims}
    
    return results

# A helper function to extract key claims
def key_claim_extraction(chunk: str) -> list:
    """
    Extracts key claims from the provided text chunk and filters irrelevant content.
    Returns a list of meaningful sentences or claims.
    """
    # Split the chunk into sentences using punctuation
    claims = re.split(r'[.?!]\s*', chunk)
    
    # Filter claims based on criteria
    filtered_claims = []
    for claim in claims:
        claim = claim.strip()
        # Skip claims that are too short or consist mainly of numbers/symbols
        if len(claim) < 16:  # Minimum length for a valid claim
            continue
        if re.match(r'^[\d%.,\s]+$', claim):  # Exclude claims with only numbers or symbols
            continue
        if not any(c.isalpha() for c in claim):  # Ensure the claim has alphabetic characters
            continue
        filtered_claims.append(claim)
    
    return filtered_claims

# from sklearn.metrics.pairwise import cosine_similarity
# from transformers import pipeline
# import spacy

# # Load models
# nlp = spacy.load("en_core_web_sm")
# sentiment_analyzer = pipeline("sentiment-analysis")

# def evaluate_chunk_consistency(article_chunk, external_data_sources):
#     """
#     Evaluates the consistency of a given article chunk.
    
#     Args:
#         article_chunk (str): The chunk of the article to evaluate.
#         external_data_sources (list): List of external sources to verify claims.

#     Returns:
#         dict: Consistency scores for internal, external, temporal, and stylistic consistency.
#     """
#     scores = {}
    
#     # Internal Consistency
#     doc = nlp(article_chunk)
#     sentences = [sent.text for sent in doc.sents]
#     sentence_embeddings = [nlp(sent).vector for sent in sentences]
#     if len(sentence_embeddings) > 1:
#         sim_matrix = cosine_similarity(sentence_embeddings)
#         avg_similarity = sim_matrix.mean()  # Average similarity between sentences
#         scores['internal_consistency'] = avg_similarity
#     else:
#         scores['internal_consistency'] = 1.0  # Single sentence is inherently consistent
    
#     # External Consistency
#     external_score = 0
#     for source in external_data_sources:
#         # Compare article chunk to external sources using cosine similarity
#         external_score += cosine_similarity([nlp(article_chunk).vector], [nlp(source).vector])[0][0]
#     scores['external_consistency'] = external_score / len(external_data_sources) if external_data_sources else 0
    
#     # Temporal Consistency
#     dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
#     # Dummy check: Ensure no repeated/conflicting dates (expand with a timeline analysis)
#     scores['temporal_consistency'] = 1.0 if len(set(dates)) == len(dates) else 0.5
    
#     # Stylistic Consistency
#     sentiments = [sentiment_analyzer(sent)[0]['label'] for sent in sentences]
#     scores['stylistic_consistency'] = 1.0 if len(set(sentiments)) == 1 else 0.5
    
#     return scores

def evaluate_consistency(chunk: str) -> str:
    # A placeholder implementation for consistency checks
    return f"Evaluating consistency within the chunk: {chunk}"

# Analyze the chunk using helper functions
def analyze_with_helper_functions(chunk: str) -> dict:
    """Analyze the chunk using helper functions."""
    # Extract key claims
    claims = key_claim_extraction(chunk)

    # Cross-check claims using the optimized context_cross_check function
    cross_check_results = context_cross_check(claims, max_results=3)

    # Evaluate consistency in the chunk
    consistency_results = evaluate_consistency(chunk)

    # Combine outputs into a dictionary
    return {
        "claims": claims,
        "cross_check_results": cross_check_results,
        "consistency_results": consistency_results,
    }

def iterative_analysis(chunk: str, full_prompt: str, new_chat_session, max_iterations: int = 3) -> dict:
    iteration_results = []
    helper_outputs = analyze_with_helper_functions(chunk)  # Call helper functions initially
    
    for i in range(max_iterations):
        prompt = f"{full_prompt}\n\n### Iteration {i+1}:\nAnalyze the text below.\n\n{chunk}\n"
        if i > 0:
            prompt += f"\nPrevious Analysis:\n{iteration_results[-1]}\n\nRefine your analysis based on the above."
        prompt += f"\nHelper Function Outputs:\n{helper_outputs}\n\n"

        try:
            response = new_chat_session.send_message(prompt)
            output = getattr(response._result, "candidates", None)
            if output and len(output) > 0:
                response_text = output[0].content.parts[0].text
                iteration_results.append(response_text)
            else:
                iteration_results.append("Error: No response.")
        except Exception as e:
            iteration_results.append(f"Error: {str(e)}")
    return iteration_results

# Analyze chunk using Google Gemini AI
def analyze_chunk_with_gemini(chunk: str) -> str:
    # Attempt to load Gemini system prompt from the text file
    try:
        with open('src/data/processed/gemini_system_prompt.txt', 'r', encoding='utf-8') as file:
            gemini_system_prompt = file.read()
    except FileNotFoundError:
        print("Error: gemini_system_prompt.txt not found.")
        return "Error: System prompt missing"
    
    # Additional task-specific instructions
    preset_prompt = (
        """
        ### Objective:
        Analyze the provided text using the following **Factuality Factors** and **Helper Functions** to detect disinformation or misinformation effectively. Perform iterative analysis across three iterations, refining the results in each pass.

        ---

        ### Factuality Factors:
        1. **Biases Factuality Factor**:
            - **Language Analysis**: Identify overt and covert language biases. Provide examples where word choices may carry inherent biases that affect interpretation.
            - **Tonal Analysis**: Assess the tone for any skew in sentiment, noting any bias towards particular topics or groups.
            - **Balanced Perspective Checks**: Evaluate if multiple perspectives are represented, especially if vital perspectives are missing or underrepresented.

        2. **Context Veracity Factor**:
            - **Consistency Checks**: Determine if the text remains consistent or contains contradictions.
            - **Contextual Shift Detection**: Detect shifts in context that could alter the meaning or interpretation of claims.
            - **Setting-based Validation**: Verify if claims are valid given the setting or situation they are presented in.

        3. **Information Utility Factor**:
            - **Content Value**: Assess whether the content provides fresh, unbiased information.
            - **Cost Analysis**: Evaluate whether there are additional barriers or costs to accessing reliable information.
            - **Reader Value**: Determine the usefulness and relevance of the content to the intended audience.

        ---

        ### Helper Functions:
        To enhance analysis, use the following helper functions during the process:
        1. **key_claim_extraction**:
            - Extract main claims or assertions from the text for focused analysis.
            - Example Output: ["Claim 1", "Claim 2", ...]
        2. **context_cross_check**:
            - Validate extracted claims against reliable external data sources.
            - Input: A list of claims and the data source (e.g., "reliable_database").
            - Example Output: { "Claim 1": "Verified", "Claim 2": "Contradicted" }
        3. **evaluate_consistency**:
            - Analyze the text for internal contradictions or logical inconsistencies.
            - Example Output: { "Consistency": "High", "Issues": ["Contradiction in paragraph 2"] }
        4. **suggest_revisions**:
            - Recommend ways to improve neutrality, consistency, or factuality of the text.
            - Example Output: ["Replace biased terminology in sentence 3", "Include a counter-perspective in paragraph 4"]

        ---

        ### Iterative Analysis Instructions:
        Perform analysis over **three iterations**, refining the results in each pass:

        1. **Iteration 1**:
            - Conduct a preliminary analysis using the Factuality Factors.
            - Identify overt and covert biases, assess tone, and check for balanced perspectives.
            - Extract key claims using `key_claim_extraction` and verify claims using `context_cross_check`.
            - Assign **preliminary scores (1 to 6)** for each micro factor.
            - **Calculate an overall score (1 to 6) for each Factuality Factor** based on the micro factor scores.

        2. **Iteration 2**:
            - Reflect on areas where the initial analysis missed nuances or misjudged factors.
            - Refine the analysis with deeper insights:
                - Reassess language for subtle biases or ambiguities.
                - Explore tonal shifts for additional layers or subtleties.
                - Check overlooked perspectives and revise the balanced perspective evaluation.
            - Use `evaluate_consistency` and `suggest_revisions` to detect gaps and improve the analysis.
            - Adjust **micro factor scores** and recalculate **overall scores**.

        3. **Iteration 3**:
            - Conduct a final review focusing on comprehensiveness:
                - Ensure diversity of perspectives is maximized.
                - Confirm that all gaps or omissions identified in earlier iterations are addressed.
                - Incorporate function outputs into the final analysis for accuracy and depth.
            - Assign **final micro factor scores (1 to 6) and an overall scaled score (1 to 6) for each Factuality Factor**.

        ---

        ### Expected JSON Output:
        Provide a structured JSON output following this format:

        ```json
        {
            "Biases_Factuality_Factor": {
                "Overall_Score": <float>,
                "Language_Analysis": {
                    "Score": <int>,
                    "Explanation": "<string>"
                },
                "Tonal_Analysis": {
                    "Score": <int>,
                    "Explanation": "<string>"
                },
                "Balanced_Perspective_Checks": {
                    "Score": <int>,
                    "Explanation": "<string>"
                }
            },
            "Context_Veracity_Factor": {
                "Overall_Score": <float>,
                "Consistency_Checks": {
                    "Score": <int>,
                    "Explanation": "<string>"
                },
                "Contextual_Shift_Detection": {
                    "Score": <int>,
                    "Explanation": "<string>"
                },
                "Setting_Based_Validation": {
                    "Score": <int>,
                    "Explanation": "<string>"
                }
            },
            "Information_Utility_Factor": {
                "Overall_Score": <float>,
                "Content_Value": {
                    "Score": <int>,
                    "Explanation": "<string>"
                },
                "Cost_Analysis": {
                    "Score": <int>,
                    "Explanation": "<string>"
                },
                "Reader_Value": {
                    "Score": <int>,
                    "Explanation": "<string>"
                }
            },
            "Overall_Score": <float>
        }
        ```

        ---
        """
    )

    # Combine system and iterative prompts
    # full_prompt = f"{gemini_system_prompt}\n\n{preset_prompt}\n\nText:\n{chunk}\n"
    # new_chat_session = model.start_chat(history=[])

    helper_outputs = analyze_with_helper_functions(chunk)
    full_prompt = f"{gemini_system_prompt}\n\n{preset_prompt}\n\nText:\n{chunk}\nHelper Function Outputs:\n{helper_outputs}\n\n"
    new_chat_session = model.start_chat(history=[])
    

    try:
        # Perform iterative analysis
        # iteration_results = iterative_analysis(chunk, full_prompt, new_chat_session)
        
        
        response = new_chat_session.send_message(full_prompt)
        output = getattr(response._result, "candidates", None)
        response_text = output[0].content.parts[0].text
        print(response_text)

        final_result = response_text
        # final_result = iteration_results[-1]

        # Step 1: Extract JSON block using regex
        json_match = re.search(r'\{.*\}', final_result, re.DOTALL)
        if not json_match:
            raise ValueError("Failed to extract JSON from AI response.")

        # Step 2: Parse JSON safely
        analysis_json = json.loads(json_match.group(0))

        return analysis_json  # Return structured dictionary
    except json.JSONDecodeError:
        print("Error: Failed to parse JSON. AI response might be malformed.")
        return {"Error": "Malformed AI response"}
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return {"Error": "Processing failed"}
  
import re

# Extract AI score from the response text
def extract_ai_score(response_text: str) -> float:
    # print(response_text)
    # Use a regular expression to find the final truthfulness score in the format of a floating-point number
    match = re.findall(r"\*{0,2}Final Truthfulness Score\:\*{0,2} ([0-9]\.[0-9]+)|\*{0,2}Final Truthfulness Score\*{0,2}\: ([0-9]\.[0-9]+)", response_text)
    result = [i for i in match[0] if len(i) > 2]
    
    if match:
        return float(result[0])  # Extract the score as a float
    else:
        return 0.0  # Return 0.0 if no score is found

# Combine the AI and predictive model scores
def combine_scores(ai_score: float, predictive_score: float, weight_ai=0.5, weight_predictive=0.5) -> float:
    return (ai_score * weight_ai) + (predictive_score * weight_predictive)

# Predictive model score function
def get_predictive_model_score(chunk_text: str) -> float:
    # Assuming `combine_feat` is a feature engineering function as in your example
    X_chunk = pd.DataFrame({'Statement': [chunk_text], 'Speaker_Job_Title': ['Unknown'], 'Extracted_Justification': ['None']})
    X_chunk = combine_feat(X_chunk)  # Apply feature engineering to get model input
    y_pred = predictive_model.predict_proba(X_chunk)
    
    # Calculate the weighted score
    weights = [0.4, 0.2, 0.6, 0.8, 0, 1]  # Example weights for each class
    weighted_data = y_pred * weights
    predictive_score = np.sum(weighted_data, axis=1)[0] # 0-6
    
    return predictive_score



# Load a pre-trained NLP model
nltk.download('punkt', quiet=True)  # Tokenizer
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
nltk.download('averaged_perceptron_tagger')  # POS tagger
nltk.download('wordnet')  # WordNet for lemmatization
nlp = spacy.load("en_core_web_sm")

# FF: content statistic

def structural_analysis(statement):
    doc = nlp(statement)
    num_sentences = len(list(doc.sents))
    total_tokens = len([token.text for token in doc])
    
    # Syntactic complexity
    avg_sentence_length = total_tokens / num_sentences if num_sentences > 0 else 0
    tree_depth = max([token.head.i - token.i for token in doc]) if len(doc) > 0 else 0
    
    # Sentiment Analysis
    sentiment = TextBlob(statement).sentiment
    polarity = sentiment.polarity
    subjectivity = sentiment.subjectivity

    return [avg_sentence_length, tree_depth, polarity, subjectivity]

def extract_graph_features(statement):
    doc = nlp(statement)
    pos_counts = Counter([token.pos_ for token in doc])
    entities = Counter([ent.label_ for ent in doc.ents])
    
    # part of speech tagging
    pos_noun = pos_counts.get("NOUN", 0)
    pos_verb = pos_counts.get("VERB", 0)
    pos_adjective = pos_counts.get("ADJ", 0)
    
    # named entity recognition
    num_persons = entities.get("PERSON", 0)
    num_orgs = entities.get("ORG", 0)
    num_gpes = entities.get("GPE", 0)
    
    return [pos_noun, pos_verb, pos_adjective, num_persons, num_orgs, num_gpes]

def extract_comparison_features(statement):
    # Keywords for different LIWC-like categories (simplified)
    cognitive_words = ["think", "know", "understand", "believe"]
    emotional_words = ["happy", "sad", "angry", "fear"]
    social_words = ["friend", "family", "society"]

    # Tokenize statement and count keywords
    vectorizer = CountVectorizer(vocabulary=cognitive_words + emotional_words + social_words)
    word_counts = vectorizer.fit_transform([statement]).toarray().flatten()

    # Divide word counts into different categories
    num_cognitive = sum(word_counts[:len(cognitive_words)])
    num_emotional = sum(word_counts[len(cognitive_words):len(cognitive_words) + len(emotional_words)])
    num_social = sum(word_counts[-len(social_words):])

    return [num_cognitive, num_emotional, num_social]

def extract_feature(statement):
    return np.array(structural_analysis(statement) + extract_graph_features(statement) + extract_comparison_features(statement))

# Function to scrape one page of fact-checks
def scrape_fact_checks(page_number):
    url = f"https://www.politifact.com/factchecks/?page={page_number}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # List to store the fact-check data
    data = []

    # Find all fact-check articles on the page
    fact_checks = soup.find_all('article', class_='m-statement')

    for fact in fact_checks:
        # Extract Author/Speaker
        author = fact.find('a', class_='m-statement__name').text.strip()

        # Extract the Date of the statement
        date_string = fact.find('div', class_='m-statement__desc').text.strip()

        # Use a regular expression to extract only the date portion (e.g., October 8, 2024)
        date_match = re.search(r'([A-Za-z]+ \d{1,2}, \d{4})', date_string)
        date = date_match.group(0) if date_match else "No date found"

        # Extract the Claim (statement being fact-checked)
        claim = fact.find('div', class_='m-statement__quote').find('a').text.strip()

        # Extract the URL to the full fact-check article
        link = "https://www.politifact.com" + fact.find('div', class_='m-statement__quote').find('a')['href']

        # Extract the Rating (e.g., False, Pants on Fire)
        rating = fact.find('div', class_='m-statement__meter').find('img')['alt'].strip()

        # Append the extracted information to the list
        data.append({
            'Author/Speaker': author,
            'Date': date,
            'Claim': claim,
            'Rating': rating,
            'Link to Full Article': link
        })

    return data

# Loop through multiple pages and collect data
def scrape_multiple_pages(start_page, end_page):
    all_data = []
    for page_number in range(start_page, end_page + 1):
        print(f"Scraping page {page_number}...")
        page_data = scrape_fact_checks(page_number)
        all_data.extend(page_data)
        time.sleep(2)  # Sleep for 2 seconds between each page request

    return all_data

# Scrape data from page 1 to 2
data = scrape_multiple_pages(1, 2)
politifact_data = pd.DataFrame(data)
test_link = politifact_data['Link to Full Article'].iloc[0]

test_response = requests.get(test_link)
soup = BeautifulSoup(test_response.text, 'html.parser')

# FF: authenticity

def cross_referenced(x, politifact_data):
    # Get the 'statement' column from the Liar Plus dataset and 'Claim' from PolitiFact
    liar_statements = x['Statement']
    politifact_claims = politifact_data['Claim']

    # Use TF-IDF Vectorizer to convert text to numerical features
    vectorizer = TfidfVectorizer(stop_words='english')

    # Combine both statements and claims for vectorization
    combined_text = pd.concat([liar_statements, politifact_claims], axis=0)

    # Fit and transform the combined text using TF-IDF
    tfidf_matrix = vectorizer.fit_transform(combined_text)

    # Split the transformed matrix into two parts: one for Liar Plus, one for PolitiFact
    liar_tfidf = tfidf_matrix[:len(liar_statements)]
    politifact_tfidf = tfidf_matrix[len(liar_statements):]

    # Compute cosine similarity between every statement in Liar Plus and every claim in PolitiFact
    similarity_matrix = cosine_similarity(liar_tfidf, politifact_tfidf)

    # Find the highest similarity score for each Liar Plus statement
    max_similarity = similarity_matrix.max(axis=1)

    # Set a threshold for similarity (e.g., 0.8) to define a "cross-referenced" statement
    threshold = 0.8
    return (max_similarity >= threshold).astype(int)

# Define a credibility score based on job title (you can customize this based on your data)
def assign_credibility_score(job_title):
    if "scientist" in job_title.lower() or "doctor" in job_title.lower():
        return 3  # High credibility
    elif "senator" in job_title.lower() or "president" in job_title.lower():
        return 2  # Medium credibility
    else:
        return 1  # Low credibility
    
# Function to detect if justification contains references to studies or data
def contains_cited_data(justification):
    keywords = ['according to', 'research', 'study', 'data', 'shown by', 'reported']
    for keyword in keywords:
        if keyword in justification.lower():
            return 1  # Cited data present
    return 0  # No cited data

sid = SentimentIntensityAnalyzer()

# Vectorizer
vectorizer = CountVectorizer(stop_words='english', max_features=50)

def fit_vectorizer(X):
    """Fit the vectorizer on the 'Statement' column of X."""
    vectorizer.fit(X['Statement'])

def transform_data(X):
    """Transform the data and generate the feature set."""
    features = []
    
    # Token/word count for normalization
    X['chunk_length'] = X['Statement'].apply(lambda x: len(x.split()))

    # Writing style and linguistic features
    X['exclamation_count'] = X['Statement'].apply(lambda x: x.count('!')) / X['chunk_length']
    X['question_mark_count'] = X['Statement'].apply(lambda x: x.count('?')) / X['chunk_length']
    X['all_caps_count'] = X['Statement'].apply(lambda x: len(re.findall(r'\b[A-Z]{2,}\b', x))) / X['chunk_length']
    
    # Named Entity Count: Counts the number of named entities in the statement using spaCy
    X['entity_count'] = X['Statement'].apply(lambda x: len(nlp(x).ents)) / X['chunk_length']
    
    # Superlative and adjective count using spaCy
    X['superlative_count'] = X['Statement'].apply(lambda x: len(re.findall(r'\b(best|worst|most|least)\b', x.lower()))) / X['chunk_length']
    X['adjective_count'] = X['Statement'].apply(lambda x: len([token for token in nlp(x) if token.pos_ == 'ADJ'])) / X['chunk_length']

    # Emotion-based word count
    emotion_words = set(['disaster', 'amazing', 'horrible', 'incredible', 'shocking', 'unbelievable'])
    X['emotion_word_count'] = X['Statement'].apply(lambda x: len([word for word in x.lower().split() if word in emotion_words])) / X['chunk_length']

    # Modal verb count
    modal_verbs = set(['might', 'could', 'must', 'should', 'would', 'may'])
    X['modal_verb_count'] = X['Statement'].apply(lambda x: len([word for word in x.lower().split() if word in modal_verbs])) / X['chunk_length']

    # Complex word ratio
    X['complex_word_ratio'] = X['Statement'].apply(lambda x: len([word for word in x.split() if len(re.findall(r'[aeiouy]+', word)) > 2]) / (len(x.split()) + 1))

    # Sentiment analysis using VADER
    X['sentiment_polarity'] = X['Statement'].apply(lambda x: sid.polarity_scores(x)['compound'])
    X['sentiment_subjectivity'] = X['Statement'].apply(lambda x: sid.polarity_scores(x)['neu'])  # Using VADER's neutrality score

    # Flesch Reading Ease
    X['flesch_reading_ease'] = X['Statement'].apply(flesch_reading_ease)

    # Combine all features into a dataframe
    numerical_features = X[['exclamation_count', 'question_mark_count', 'all_caps_count',
                            'entity_count', 'superlative_count', 'adjective_count',
                            'emotion_word_count', 'modal_verb_count', 
                            'complex_word_ratio', 
                            'sentiment_polarity', 'sentiment_subjectivity', 'flesch_reading_ease']]
    
    features.append(numerical_features)
    return pd.concat(features, axis=1)

def flesch_reading_ease(text):
    """Compute Flesch Reading Ease score for readability analysis."""
    sentence_count = max(len(re.split(r'[.!?]+', text)), 1)
    word_count = len(text.split())
    syllable_count = sum([syllable_count_func(word) for word in text.split()])
    if word_count > 0:
        return (206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllable_count / word_count)) / 100
    else:
        return 0

def syllable_count_func(word):
    """Count syllables in a word using regular expressions."""
    word = word.lower()
    syllables = re.findall(r'[aeiouy]+', word)
    return max(len(syllables), 1)

# Combine all the engineered features
def combine_feat(X):
    X['cross_referenced'] = cross_referenced(X, politifact_data)  # Cross-referencing feature
    X['credibility_score'] = X['Speaker_Job_Title'].apply(assign_credibility_score)  # Author credentials feature
    X['cited_data'] = X['Extracted_Justification'].apply(contains_cited_data)  # Cited data verification feature
    X_auth = X[['cross_referenced', 'credibility_score', 'cited_data']].to_numpy()
    X_content = np.vstack(X['Statement'].apply(lambda x: extract_feature(x)).to_numpy())
    fit_vectorizer(X)
    X_ling_tox = transform_data(X).to_numpy()
    X = np.hstack((X_content, X_auth, X_ling_tox))
    return X


# FF: content statistic

def structural_analysis(statement):
    doc = nlp(statement)
    num_sentences = len(list(doc.sents))
    total_tokens = len([token.text for token in doc])
    
    # Syntactic complexity
    avg_sentence_length = total_tokens / num_sentences if num_sentences > 0 else 0
    tree_depth = max([token.head.i - token.i for token in doc]) if len(doc) > 0 else 0
    
    # Sentiment Analysis
    sentiment = TextBlob(statement).sentiment
    polarity = sentiment.polarity
    subjectivity = sentiment.subjectivity

    return [avg_sentence_length, tree_depth, polarity, subjectivity]

def extract_graph_features(statement):
    doc = nlp(statement)
    pos_counts = Counter([token.pos_ for token in doc])
    entities = Counter([ent.label_ for ent in doc.ents])
    
    # part of speech tagging
    pos_noun = pos_counts.get("NOUN", 0)
    pos_verb = pos_counts.get("VERB", 0)
    pos_adjective = pos_counts.get("ADJ", 0)
    
    # named entity recognition
    num_persons = entities.get("PERSON", 0)
    num_orgs = entities.get("ORG", 0)
    num_gpes = entities.get("GPE", 0)
    
    return [pos_noun, pos_verb, pos_adjective, num_persons, num_orgs, num_gpes]

def extract_comparison_features(statement):
    # Keywords for different LIWC-like categories (simplified)
    cognitive_words = ["think", "know", "understand", "believe"]
    emotional_words = ["happy", "sad", "angry", "fear"]
    social_words = ["friend", "family", "society"]

    # Tokenize statement and count keywords
    vectorizer = CountVectorizer(vocabulary=cognitive_words + emotional_words + social_words)
    word_counts = vectorizer.fit_transform([statement]).toarray().flatten()

    # Divide word counts into different categories
    num_cognitive = sum(word_counts[:len(cognitive_words)])
    num_emotional = sum(word_counts[len(cognitive_words):len(cognitive_words) + len(emotional_words)])
    num_social = sum(word_counts[-len(social_words):])

    return [num_cognitive, num_emotional, num_social]

def compute_factuality_score(statement):
    # Extract features
    struct_features = structural_analysis(statement)
    graph_features = extract_graph_features(statement)
    comp_features = extract_comparison_features(statement)

    all_features = np.array(struct_features + graph_features + comp_features)

    # Apply Rescaling to Large Features
    rescaled_features = np.array([
        all_features[0] / 50,  # Sentence length
        all_features[1] / 10,  # Tree depth
        all_features[2],       # Polarity
        all_features[3],       # Subjectivity

        all_features[4] / 10,  # POS Nouns
        all_features[5] / 10,  # POS Verbs
        all_features[6] / 10,  # POS Adjectives
        all_features[7] / 5,   # Named Entities - Persons
        all_features[8] / 5,   # Named Entities - Organizations
        all_features[9] / 5,   # Named Entities - GPE

        all_features[10],      # Cognitive words
        all_features[11],      # Emotional words
        all_features[12],      # Social words
    ])

    # Adjusted Weights
    weights = np.array([
        0.15, 0.15, 0.10, 0.10,  # Structural: length, depth, polarity, subjectivity
        0.10, 0.10, 0.10, 0.05, 0.05, 0.04,  # Graph: POS, Entities
        0.02, 0.02, 0.02  # Comparison: Cognitive, Emotional, Social words
    ])

    factuality_score = np.dot(rescaled_features, weights)
    print(factuality_score)

    # Convert to 1-6 scale
    rating = round(min(max(factuality_score * 5 + 1, 1), 6), 1)

    return rating

def extract_feature(statement):
    return np.array(structural_analysis(statement) + extract_graph_features(statement) + extract_comparison_features(statement))

# Function to scrape one page of fact-checks
def scrape_fact_checks(page_number):
    url = f"https://www.politifact.com/factchecks/?page={page_number}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # List to store the fact-check data
    data = []

    # Find all fact-check articles on the page
    fact_checks = soup.find_all('article', class_='m-statement')

    for fact in fact_checks:
        # Extract Author/Speaker
        author = fact.find('a', class_='m-statement__name').text.strip()

        # Extract the Date of the statement
        date_string = fact.find('div', class_='m-statement__desc').text.strip()

        # Use a regular expression to extract only the date portion (e.g., October 8, 2024)
        date_match = re.search(r'([A-Za-z]+ \d{1,2}, \d{4})', date_string)
        date = date_match.group(0) if date_match else "No date found"

        # Extract the Claim (statement being fact-checked)
        claim = fact.find('div', class_='m-statement__quote').find('a').text.strip()

        # Extract the URL to the full fact-check article
        link = "https://www.politifact.com" + fact.find('div', class_='m-statement__quote').find('a')['href']

        # Extract the Rating (e.g., False, Pants on Fire)
        rating = fact.find('div', class_='m-statement__meter').find('img')['alt'].strip()

        # Append the extracted information to the list
        data.append({
            'Author/Speaker': author,
            'Date': date,
            'Claim': claim,
            'Rating': rating,
            'Link to Full Article': link
        })

    return data

# Loop through multiple pages and collect data
def scrape_multiple_pages(start_page, end_page):
    all_data = []
    for page_number in range(start_page, end_page + 1):
        print(f"Scraping page {page_number}...")
        page_data = scrape_fact_checks(page_number)
        all_data.extend(page_data)
        time.sleep(2)  # Sleep for 2 seconds between each page request

    return all_data

# Scrape data from page 1 to 2
data = scrape_multiple_pages(1, 2)
politifact_data = pd.DataFrame(data)
test_link = politifact_data['Link to Full Article'].iloc[0]

test_response = requests.get(test_link)
soup = BeautifulSoup(test_response.text, 'html.parser')

# FF: authenticity

def cross_referenced(x, politifact_data):
    # Get the 'statement' column from the Liar Plus dataset and 'Claim' from PolitiFact
    liar_statements = x['Statement']
    politifact_claims = politifact_data['Claim']

    # Use TF-IDF Vectorizer to convert text to numerical features
    vectorizer = TfidfVectorizer(stop_words='english')

    # Combine both statements and claims for vectorization
    combined_text = pd.concat([liar_statements, politifact_claims], axis=0)

    # Fit and transform the combined text using TF-IDF
    tfidf_matrix = vectorizer.fit_transform(combined_text)

    # Split the transformed matrix into two parts: one for Liar Plus, one for PolitiFact
    liar_tfidf = tfidf_matrix[:len(liar_statements)]
    politifact_tfidf = tfidf_matrix[len(liar_statements):]

    # Compute cosine similarity between every statement in Liar Plus and every claim in PolitiFact
    similarity_matrix = cosine_similarity(liar_tfidf, politifact_tfidf)

    # Find the highest similarity score for each Liar Plus statement
    max_similarity = similarity_matrix.max(axis=1)

    # Set a threshold for similarity (e.g., 0.8) to define a "cross-referenced" statement
    threshold = 0.8
    return (max_similarity >= threshold).astype(int)



# Define a credibility score based on job title (you can customize this based on your data)
def assign_credibility_score(job_title):
    if "scientist" in job_title.lower() or "doctor" in job_title.lower():
        return 3  # High credibility
    elif "senator" in job_title.lower() or "president" in job_title.lower():
        return 2  # Medium credibility
    else:
        return 1  # Low credibility



# Function to detect if justification contains references to studies or data
def contains_cited_data(justification):
    keywords = ['according to', 'research', 'study', 'data', 'shown by', 'reported']
    for keyword in keywords:
        if keyword in justification.lower():
            return 1  # Cited data present
    return 0  # No cited data

def compute_authenticity_score(statement, job_title, justification, politifact_data):
    # Compute cross-referenced similarity (Binary: 0 or 1)
    cross_ref_score = cross_referenced(pd.DataFrame({'Statement': [statement]}), politifact_data)[0]

    # Assign credibility score based on job title (1, 2, or 3) → Scale to [0,1]
    raw_credibility_score = assign_credibility_score(job_title)  # 1, 2, or 3
    credibility_score = raw_credibility_score / 3  # Normalize to [0,1]

    # Check if justification contains cited data (Binary: 0 or 1)
    cited_data_score = contains_cited_data(justification)

    # Feature Vector (All in range [0,1])
    feature_vector = np.array([cross_ref_score, credibility_score, cited_data_score])

    # Adjusted Weights
    weights = np.array([0.20, 0.60, 0.20])

    # Compute weighted authenticity score
    authenticity_score = np.dot(feature_vector, weights)
    print(authenticity_score)

    # Convert to 1-6 scale
    rating = round(min(max((authenticity_score * 5) + 1, 1), 6), 1)  # Adjusted scaling

    return rating

## Linguistic based

sid = SentimentIntensityAnalyzer()

# Vectorizer
vectorizer = CountVectorizer(stop_words='english', max_features=50)

def fit_vectorizer(X):
    """Fit the vectorizer on the 'Statement' column of X."""
    vectorizer.fit(X['Statement'])

def transform_data(X):
    """Transform the data and generate the feature set."""
    features = []
    
    # Token/word count for normalization
    X['chunk_length'] = X['Statement'].apply(lambda x: len(x.split()))

    # Writing style and linguistic features
    X['exclamation_count'] = X['Statement'].apply(lambda x: x.count('!')) / X['chunk_length']
    X['question_mark_count'] = X['Statement'].apply(lambda x: x.count('?')) / X['chunk_length']
    X['all_caps_count'] = X['Statement'].apply(lambda x: len(re.findall(r'\b[A-Z]{2,}\b', x))) / X['chunk_length']
    
    # Named Entity Count: Counts the number of named entities in the statement using spaCy
    X['entity_count'] = X['Statement'].apply(lambda x: len(nlp(x).ents)) / X['chunk_length']
    
    # Superlative and adjective count using spaCy
    X['superlative_count'] = X['Statement'].apply(lambda x: len(re.findall(r'\b(best|worst|most|least)\b', x.lower()))) / X['chunk_length']
    X['adjective_count'] = X['Statement'].apply(lambda x: len([token for token in nlp(x) if token.pos_ == 'ADJ'])) / X['chunk_length']

    # Emotion-based word count
    emotion_words = set(['disaster', 'amazing', 'horrible', 'incredible', 'shocking', 'unbelievable'])
    X['emotion_word_count'] = X['Statement'].apply(lambda x: len([word for word in x.lower().split() if word in emotion_words])) / X['chunk_length']

    # Modal verb count
    modal_verbs = set(['might', 'could', 'must', 'should', 'would', 'may'])
    X['modal_verb_count'] = X['Statement'].apply(lambda x: len([word for word in x.lower().split() if word in modal_verbs])) / X['chunk_length']

    # Complex word ratio
    X['complex_word_ratio'] = X['Statement'].apply(lambda x: len([word for word in x.split() if len(re.findall(r'[aeiouy]+', word)) > 2]) / (len(x.split()) + 1))

    # Sentiment analysis using VADER
    X['sentiment_polarity'] = X['Statement'].apply(lambda x: sid.polarity_scores(x)['compound'])
    X['sentiment_subjectivity'] = X['Statement'].apply(lambda x: sid.polarity_scores(x)['neu'])  # Using VADER's neutrality score

    # Flesch Reading Ease
    X['flesch_reading_ease'] = X['Statement'].apply(flesch_reading_ease)

    # Combine all features into a dataframe
    numerical_features = X[['exclamation_count', 'question_mark_count', 'all_caps_count',
                            'entity_count', 'superlative_count', 'adjective_count',
                            'emotion_word_count', 'modal_verb_count', 
                            'complex_word_ratio', 
                            'sentiment_polarity', 'sentiment_subjectivity', 'flesch_reading_ease']]
    
    features.append(numerical_features)
    return pd.concat(features, axis=1)

def flesch_reading_ease(text):
    """Compute Flesch Reading Ease score for readability analysis."""
    sentence_count = max(len(re.split(r'[.!?]+', text)), 1)
    word_count = len(text.split())
    syllable_count = sum([syllable_count_func(word) for word in text.split()])
    if word_count > 0:
        return (206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllable_count / word_count)) / 100
    else:
        return 0

def syllable_count_func(word):
    """Count syllables in a word using regular expressions."""
    word = word.lower()
    syllables = re.findall(r'[aeiouy]+', word)
    return max(len(syllables), 1)

# Load SpaCy NLP model
nlp = spacy.load("en_core_web_sm")
sid = SentimentIntensityAnalyzer()

def compute_linguistic_persuasion_score(statement):
    """
    Computes a persuasion/exaggeration score based on a single statement.
    """

    # Token/word count for normalization
    chunk_length = max(len(statement.split()), 1)

    # Writing style and linguistic features
    exclamation_count = (statement.count('!') / chunk_length) * 10  # Increased scaling
    question_mark_count = (statement.count('?') / chunk_length) * 10  
    all_caps_count = (len(re.findall(r'\b[A-Z]{2,}\b', statement)) / chunk_length) * 6  

    # Named Entity Count
    doc = nlp(statement)
    entity_count = (len(doc.ents) / chunk_length) * 3  # Scaled up

    # Superlative and adjective count
    superlative_count = (len(re.findall(r'\b(best|worst|most|least)\b', statement.lower())) / chunk_length) * 8  
    adjective_count = (len([token for token in doc if token.pos_ == 'ADJ']) / chunk_length) * 2  

    # Emotion-based words
    emotion_words = {'disaster', 'amazing', 'horrible', 'incredible', 'shocking', 'unbelievable'}
    emotion_word_count = (len([word for word in statement.lower().split() if word in emotion_words]) / chunk_length) * 7  

    # Modal verb count
    modal_verbs = {'might', 'could', 'must', 'should', 'would', 'may'}
    modal_verb_count = (len([word for word in statement.lower().split() if word in modal_verbs]) / chunk_length) * 5  

    # Complex word ratio
    complex_word_ratio = len([word for word in statement.split() if len(re.findall(r'[aeiouy]+', word)) > 2]) / chunk_length  

    # Sentiment analysis using VADER
    sentiment_polarity = abs(sid.polarity_scores(statement)['compound'])  # Stronger absolute polarity affects factuality
    sentiment_subjectivity = 1 - sid.polarity_scores(statement)['neu']  # Higher non-neutrality affects factuality

    def flesch_reading_ease(text):
        """Compute Flesch Reading Ease score for readability analysis."""
        sentence_count = max(len(re.split(r'[.!?]+', text)), 1)
        word_count = len(text.split())

        def syllable_count_func(word):
            """Count syllables in a word using regex."""
            word = word.lower()
            syllables = re.findall(r'[aeiouy]+', word)
            return max(len(syllables), 1)
        
        syllable_count = sum([syllable_count_func(word) for word in text.split()])
        if word_count > 0:
            return (206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllable_count / word_count)) / 100
        else:
            return 0
    
    # Readability score (Flesch Reading Ease)
    flesch_reading_ease = flesch_reading_ease(statement) / 100  # Scale it to [0,1]

    # Apply **non-linear transformation** to exaggeration & sentiment
    exaggeration_score = (
        exclamation_count**1.2 + 
        question_mark_count**1.2 + 
        all_caps_count**1.2 + 
        superlative_count**1.2 + 
        emotion_word_count**1.2 + 
        modal_verb_count**1.2
    ) / 6  

    sentiment_score = (sentiment_polarity**1.5 + sentiment_subjectivity**1.5 + flesch_reading_ease) / 3  

    linguistic_complexity = (entity_count - adjective_count + complex_word_ratio) / 3  

    # Feature Vector
    feature_vector = np.array([
        exaggeration_score,  
        linguistic_complexity,  
        sentiment_score  
    ])

    print("Feature values:", feature_vector)  # Debugging

    # Adjusted Weights
    weights = np.array([0.55, 0.25, 0.20])

    # Compute weighted persuasion score
    persuasion_score = np.dot(feature_vector, weights)

    # **Adjust scaling to create more spread in scores**
    adjusted_score = np.tanh(persuasion_score)  # Apply **non-linear scaling**
    print(adjusted_score)
    rating = round(min(max(((adjusted_score) * 5) + 1, 1), 6)  , 1)

    return rating

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))  # Default port to 8080 if Render does not set PORT
    me.run(port=port, host="0.0.0.0")  # Ensure the app binds to the correct host and port