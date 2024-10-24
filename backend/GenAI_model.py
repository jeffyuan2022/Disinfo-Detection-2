import os
import mesop as me
from PyPDF2 import PdfReader
import google.generativeai as genai
import io

GOOGLE_GENAI_API_KEY = os.getenv("GOOGLE_GENAI_API_KEY")

# Gemini API setup
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

@me.stateclass
class State:
    file: me.UploadedFile = None  # Initialize with None
    pdf_content: str = ""  # Store PDF content
    analysis_result: str = ""  # Store analysis result
    is_loading: bool = False  # To indicate loading

def load(e: me.LoadEvent):
    me.set_theme_mode("system")

class App:
    @me.page(
        on_load=load,
        security_policy=me.SecurityPolicy(
            allowed_iframe_parents=["https://google.github.io"]
        ),
        path="/",
    )
    def app():  # No parameters needed
        state = me.state(State)
        with me.box(style=me.Style(padding=me.Padding.all(15))):
            me.uploader(
                label="Upload PDF",
                accepted_file_types=["application/pdf"],  # Accept only PDFs
                on_upload=handle_upload,
                type="flat",
                color="primary",
                style=me.Style(font_weight="bold"),
            )

            if state.file and state.file.size:
                with me.box(style=me.Style(margin=me.Margin.all(10))):
                    me.text(f"File name: {state.file.name}")
                    me.text(f"File size: {state.file.size}")
                    me.text(f"File type: {state.file.mime_type}")

                with me.box(style=me.Style(margin=me.Margin.all(10))):
                    me.text(f"Extracted PDF Content (first 500 chars): {state.pdf_content[:500]}...")

                # Trigger the analysis with Gemini
                me.button(label="Analyze PDF", on_click=lambda: analyze_pdf_with_gemini(state.pdf_content))

            if state.is_loading:
                with me.box(style=me.Style(margin=me.Margin.all(10))):
                    me.text("Analyzing... Please wait.")

            elif state.analysis_result:
                with me.box(style=me.Style(margin=me.Margin.all(10))):
                    me.text(f"Analysis Result: {state.analysis_result}")
            else:
                with me.box(style=me.Style(margin=me.Margin.all(10))):
                    me.text("Upload a PDF and click 'Analyze PDF' to get a result.")

def handle_upload(event: me.UploadEvent):
    state = me.state(State)
    state.file = event.file
    if event.file.mime_type == "application/pdf":
        state.pdf_content = extract_pdf_content(event.file)

def extract_pdf_content(file: me.UploadedFile) -> str:
    """Extracts text from an uploaded PDF file."""
    file_like_object = io.BytesIO(file.getvalue())
    
    reader = PdfReader(file_like_object)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    
    return text

def analyze_pdf_with_gemini(pdf_text: str):
    """Analyze the uploaded PDF content using the Gemini API."""
    state = me.state(State)

    # Show loading indicator
    state.is_loading = True

    try:
        # Generate the prompt for analysis
        full_input = generate_truth_analysis_prompt(pdf_text)
        
        # Start a chat session with the Gemini model
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(full_input)
        
        # Store the analysis result in the state
        state.analysis_result = response.text
    except Exception as e:
        # Handle potential errors
        state.analysis_result = f"Error during analysis: {str(e)}"
    finally:
        # Remove loading indicator
        state.is_loading = False

    # Ensure the state gets updated and triggers a re-render
    me.update_state(state)

def generate_truth_analysis_prompt(pdf_text: str) -> str:
    """Generates a prompt to ask the Gemini model to analyze truthfulness of the PDF content."""
    return (
        f"Analyze the following document and provide an assessment of its truthfulness. "
        f"Score the document from 0 to 10, where 0 means completely false, and 10 means completely true. "
        f"Provide detailed reasons for the score you assign: \n\n{pdf_text}"
    )

# Create an instance of the App class
app = App()