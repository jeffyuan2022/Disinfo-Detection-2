import PyPDF2
import mesop as me
import mesop.labs as mel
import google.generativeai as genai
import os
from dotenv import load_dotenv
from typing import List

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_GENAI_API_KEY"))

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

# response = chat_session.send_message("What can you do?")

# output = getattr(response._result, "candidates", None)[0].content.parts[0].text
# print(output)

# File upload setup in Mesop
@me.page(
    security_policy=me.SecurityPolicy(
        allowed_iframe_parents=["https:google.github.io"]
    ),
    path="/",
    title= "Mesop PDF Analysis",
)

def page():
    mel.chat(transform, title="Gemini Chat Bot")

def transform(prompt: str, history: list[mel.ChatMessage]):
    chat_session = model.start_chat(history=[])
    try:
        # Send message and get response
        response = chat_session.send_message(prompt)
        output = getattr(response._result, "candidates", None)
        if output and len(output) > 0:
            # Extract the first candidate's first part of the text
            response_text = output[0].content.parts[0].text
            yield response_text
        else:
            yield "No response from the model."
            
    except Exception as e:
        yield f"Error processing response: {str(e)}"
