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
    Analyze the provided text using the following **Factuality Factors**. Each factor has three mini-factors, and each factor should be scored from 0 to 1 (2 decimal places). After the analysis, generate an overall **Final Truthfulness Score** from 0 to 1, where 0 represents 0% truth and 1 represents 100% truth. A brief explanation for each factor is required. Format your response as follows:

    ### **1. Biases Factuality Factor**:
    - **Language Analysis Score**: Provide a score between 0 and 1.
    - Explanation: Brief explanation of how language bias (overt or covert) is detected or absent.
    
    - **Tonal Analysis Score**: Provide a score between 0 and 1.
    - Explanation: Brief explanation of how the tone affects the neutrality or bias of the text.

    - **Balanced Perspective Checks Score**: Provide a score between 0 and 1.
    - Explanation: Brief explanation of whether multiple perspectives are fairly represented.

    ### **2. Context Veracity Factor**:
    - **Consistency Checks Score**: Provide a score between 0 and 1.
    - Explanation: Brief explanation of whether the content remains consistent or has contradictions.

    - **Contextual Shift Detection Score**: Provide a score between 0 and 1.
    - Explanation: Brief explanation of any shifts in context that could alter the meaning of the text.

    - **Setting-based Validation Score**: Provide a score between 0 and 1.
    - Explanation: Brief explanation of whether the claims are valid based on the setting or situation they are presented in.

    ### **3. Information Utility Factor**:
    - **Content Value Score**: Provide a score between 0 and 1.
    - Explanation: Brief explanation of whether the content provides fresh, unbiased information.

    - **Cost Analysis Score**: Provide a score between 0 and 1.
    - Explanation: Brief explanation of whether there are additional costs or barriers to accessing reliable information.

    - **Reader Value Score**: Provide a score between 0 and 1.
    - Explanation: Brief explanation of how useful the content is to the intended audience.

    ### **Final Truthfulness Score**:
    - Based on the above factor scores, calculate the **Final Truthfulness Score** between 0 and 1 (2 decimal places) that represents the overall truthfulness of the text chunk.
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
