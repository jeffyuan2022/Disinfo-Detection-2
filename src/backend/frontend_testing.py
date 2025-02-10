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


def page():
  with me.box(
    style=me.Style(
      background="#fff",
      min_height="calc(100% - 48px)",
      padding=me.Padding(bottom=16),
    )
  ):
    with me.box(
      style=me.Style(
        width="min(720px, 100%)",
        margin=me.Margin.symmetric(horizontal="auto"),
        padding=me.Padding.symmetric(
          horizontal=16,
        ),
      )
    ):
     header_text()
     app()
    footer()


def header_text():
  with me.box(
    style=me.Style(
      position="absolute",
      padding=me.Padding.symmetric(vertical=45, horizontal=45),
      width="100%",
      background= "#e9f1fe",
      color = "#4285F4",
      left = 0,
      right = 0
    )
  ):
    me.text(
      "Misinformation & Disinformation Detection",
      style=me.Style(
        font_size = 35,
        font_weight=1000
      ),
    )

def footer():
  with me.box(
    style=me.Style(
      position="absolute",
      bottom=0,
      padding=me.Padding.symmetric(vertical=16, horizontal=16),
      width="100%",
      background= "#e9f1fe",
      color = "#4285F4",
      font_size=14,
    )
  ):
    me.html(
      "GenAI for Good 2025 DSC Capstoe Project: <a href='https://github.com/jeffyuan2022/Disinfo-Detection-2/tree/main'>Github Link</a>",
    )

def app():
  state = me.state(State)
  with me.box(style=me.Style(padding=me.Padding.all(200))):
    with me.box(style=me.Style(display="flex", gap=10)):
      
      #insert image
      me.image(
        src="/fake_news.png",
        alt="Fake News",
        style=me.Style(width="100%"),
      ),

      # URL Input Section
      me.input(
          label="Enter Article URL",
          placeholder="https://example.com/article",
          on_blur=handle_url_input,
          appearance="outline",
          style=me.Style(
            font_weight="bold",
            font_size="25px",  
            padding=me.Padding.symmetric(vertical=20, horizontal=20),  
            width="100vw",  
            max_width="220px",  
            height="60px",  
            text_align="center",
            position="absolute",  
            top="25%",  
            left="20%",  
            transform="translate(-50%, -50%)",  )
      )
      

      me.button(
        label="Scrape Article",
        on_click=handle_scrape_article,
        type="flat",
        color="primary",
        style=me.Style(
            font_weight="bold",
            font_size="20px",  
            padding=me.Padding.symmetric(vertical=20, horizontal=20),  
            width="100vw",  
            max_width="220px",  
            height="60px",  
            text_align="center",
            position="absolute",  
            top="27.5%",  
            left="40%",  
            transform="translate(-50%, -50%)",  )
      )

          
      # PDF Uploader
      me.uploader(
        label="Upload PDF File",
        accepted_file_types=["application/pdf"],
        on_upload=handle_upload,
        type="raised",
        color="primary",
        style=me.Style(
            font_weight="bold", 
            font_size="20px",  
            padding=me.Padding.symmetric(vertical=20, horizontal=20),  
            width="100vw",  
            max_width="220px",  
            height="60px",  
            text_align="center",
            position="absolute",  
            top="27.5%",  
            left="60%",  
            transform="translate(-50%, -50%)", 
        ),
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
