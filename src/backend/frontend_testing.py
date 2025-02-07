import mesop as me
from typing import List

## test push

@me.stateclass
class State:
    file: me.UploadedFile
    url: str = ""
    url_article: str = ""

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
    with me.box(style=me.Style(display="flex", gap=20)):
      # URL Input Section
      me.input(
          label="Enter Article URL",
          placeholder="https://example.com/article",
          on_blur=handle_url_input,
          appearance="outline",
      )
      me.button(
        label="Scrape Article",
        on_click=handle_scrape_article,
        type="flat",
        color="primary",
        style=me.Style(font_weight="bold"),
      )

      # PDF Uploader
      me.uploader(
        label="Upload PDF File",
        accepted_file_types=["application/pdf"],
        on_upload=handle_upload,
        type="raised",
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
        for result in analysis_results:
          me.text(f"{result}")

      # Display results for URL scraping
      if state.url_article:
        with me.box(style=me.Style(margin=me.Margin.all(10))):
          me.text("Scraped Article:")
          me.text(state.url_article)
          analysis_results = analyze_article_text(state.url_article)
          for result in analysis_results:
            me.text(f"{result}")

def handle_upload(event: me.UploadEvent):
  state = me.state(State)
  state.file = event.file

# Store the URL entered by the user
def handle_url_input(event: me.InputBlurEvent):
    state = me.state(State)
    state.input = event.value

# Scrape article from the provided URL
def handle_scrape_article(event: me.ClickEvent):
    state = me.state(State)
    if state.input:
        try:
            state.url_article = scrape_article_from_url(state.url)
        except Exception as e:
            state.url_article = f"Failed to scrape article: {e}"
    else:
        state.url_article = "Please enter a valid URL."

# Scrape article content from the URL
def scrape_article_from_url(url: str) -> str:
    import requests
    from bs4 import BeautifulSoup

    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    article = ""
    for paragraph in soup.find_all("p"):
        article += paragraph.get_text() + "\n"

    return article.strip()

def analyze_article_text(article_text: str) -> List[str]:
    return True

# Update process_pdf_and_analyze to use Weaviate for chunk storage and retrieval
def process_pdf_and_analyze(file: me.UploadedFile) -> List[str]: 
    return True
