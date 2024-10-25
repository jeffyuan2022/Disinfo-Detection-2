import base64
import PyPDF2
import mesop as me
import google.generativeai as genai
from typing import List
import os
from io import BytesIO
from dotenv import load_dotenv

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
  file: me.UploadedFile

def load(e: me.LoadEvent):
  me.set_theme_mode("system")

@me.page(
  on_load=load,
  security_policy=me.SecurityPolicy(
    allowed_iframe_parents=["https://google.github.io"]
  ),
  path="/uploader",
)
def app():
  state = me.state(State)
  
  with me.box(style=me.Style(padding=me.Padding.all(15))):
    # PDF Uploader
    me.uploader(
      label="Upload PDF File",
      accepted_file_types=["application/pdf"],
      on_upload=handle_upload,
      type="flat",
      color="primary",
      style=me.Style(font_weight="bold"),
    )

    # If file is uploaded, show file details and process it
    if state.file.size:
      with me.box(style=me.Style(margin=me.Margin.all(10))):
        me.text(f"File name: {state.file.name}")
        me.text(f"File size: {state.file.size}")
        me.text(f"File type: {state.file.mime_type}")

      # Process PDF and display results
      analysis_results = process_pdf_and_analyze(state.file)
      for i, result in enumerate(analysis_results):
        me.text(f"Chunk {i+1} score: {result}")


def handle_upload(event: me.UploadEvent):
  state = me.state(State)
  state.file = event.file

def process_pdf_and_analyze(file: me.UploadedFile) -> List[str]:
  # Step 1: Extract text from the uploaded PDF file
  pdf_text = extract_text_from_pdf(file)
  
  # Step 2: Chunk the text into smaller pieces
  chunks = chunk_text(pdf_text)
  
  # Step 3: Analyze each chunk using the preset prompt
  results = []
  for chunk in chunks:
    score = analyze_chunk_with_gemini(chunk)
    results.append(score)
  
  return results

def extract_text_from_pdf(file: me.UploadedFile) -> str:
    # Wrap the bytes in a BytesIO stream for PyPDF2
    pdf_stream = BytesIO(file.getvalue())
    reader = PyPDF2.PdfReader(pdf_stream)
    
    extracted_text = ""
    for page in reader.pages:
        extracted_text += page.extract_text()
    
    return extracted_text

def chunk_text(text: str, chunk_size=2000) -> List[str]:
  # Split the text into smaller chunks
  words = text.split()
  chunks = []
  current_chunk = []
  current_chunk_length = 0

  for word in words:
    current_chunk.append(word)
    current_chunk_length += len(word) + 1  # +1 for space

    if current_chunk_length > chunk_size:
      chunks.append(" ".join(current_chunk))
      current_chunk = []
      current_chunk_length = 0

  if current_chunk:  # Add any remaining text as the last chunk
    chunks.append(" ".join(current_chunk))

  return chunks

def analyze_chunk_with_gemini(chunk: str) -> str:
  # Preset prompt for the Biases Factuality Factor
  new_chat_session = model.start_chat(history=[])
  preset_prompt = (
    """
    Analyze the provided text based on the **Biases Factuality Factor**, and give it a score from 0 to 10. Provide detailed analysis across the following three areas, with specific strengths and weaknesses for each:

    1. **Language Analysis** (Score out of 10): 
    - Detect any overt or covert language biases. Is the language neutral, or does it subtly push a particular viewpoint? Highlight any loaded or suggestive phrases that might influence readers, even if factual.

    2. **Tonal Analysis** (Score out of 10): 
    - Evaluate whether the tone of the text is neutral or if it leans disproportionately positive or negative toward certain topics or groups. Does the tone contribute to bias or provide a balanced presentation of facts?

    3. **Balanced Perspective Checks** (Score out of 10): 
    - Ensure that multiple perspectives are fairly represented. Does the text offer a complete view of the topic, or does it omit vital perspectives? Highlight areas where the text could have provided a more balanced representation of opposing views.

    For each of the three areas, clearly explain the **strengths** and **weaknesses** that contribute to your score, providing specific examples from the text. Summarize your overall impression of the text and its potential biases.
    """
  )
  full_prompt = f"{preset_prompt}\n\nText:\n{chunk}"

  try:
    response = new_chat_session.send_message(full_prompt)
    # Extract response text (as per the working structure)
    output = getattr(response._result, "candidates", None)
    if output and len(output) > 0:
      response_text = output[0].content.parts[0].text
      return response_text
    else:
      return "No response from the model."
  except Exception as e:
    return f"Error processing chunk: {str(e)}"
