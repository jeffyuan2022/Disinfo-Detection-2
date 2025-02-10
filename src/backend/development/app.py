# app.py
import mesop as me
import article_extractor


@me.stateclass
class State:
  url: str = ""
  article_title: str = ""
  article_text: str = ""
  error_message: str = ""
  loading: bool = False


def extract_action(url: str):
  state = me.state(State)
  state.loading = True
  state.article_title = ""
  state.article_text = ""
  state.error_message = ""

  result = article_extractor.extract_article(url)

  state.loading = False
  if result["error"]:
    state.error_message = result["error"]
  else:
    state.article_title = result["title"]
    state.article_text = result["text"]


@me.page(path="/")
def main():
  state = me.state(State)

  with me.box(style={
      "display": "flex",
      "flex-direction": "column",
      "gap": "16px",
  }):
    me.text(text="Article Extractor", type="headline-1")
    with me.box(style={
        "display": "flex",
        "flex-direction": "row",
        "gap": "8px",
        "align-items": "center",
    }):
      me.input(
          label="Enter Article URL",
          on_input=lambda event: extract_action(event.value),
          style={"width": "400px"},
          value=state.url,
      )
      me.button(
          on_click=lambda: extract_action(state.url),
          label="Extract",  # Changed 'text' to 'label'
          disabled=not state.url,
      )

    if state.loading:
      me.progress_spinner(style={"margin-top": "16px"})

    if state.error_message:
      me.text(text=f"Error: {state.error_message}", type="headline-4", style={"color": "red"})

    if state.article_title:
      me.text(text=state.article_title, type="headline-2")
    if state.article_text:
      me.textarea(value=state.article_text, readonly=True, rows=20, style={"width": "80%"})