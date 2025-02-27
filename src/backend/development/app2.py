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
def app():
  state = me.state(State)

  with me.box(style=me.Style(max_width=1000, margin=me.Margin.symmetric(horizontal="auto"), padding=me.Padding.all(24), font_family="sans-serif", display="flex", flex_direction="column", min_height="100vh")):
    me.text(text="Article Extractor", type="headline-2")
    
    with me.box(style=me.Style(display="flex", flex_direction="column", gap=8)):
      me.input(
          label="Enter Article URL",
          on_input=lambda event: extract_action(event.value),
          style=me.Style(
              width="100%",
              font_size=16,
              padding=me.Padding.all(8),
              border=me.Border.all(me.BorderSide(width=1, color="#ccc", style="solid")),
              border_radius=4
          ),
          value=state.url,
      )
      me.button(
          on_click=lambda: extract_action(state.url),
          label="Extract",
          disabled=not state.url,
          style=me.Style(
              background="#007bff",
              color="white",
              border=me.Border.all(me.BorderSide(width=0)),
              padding=me.Padding.symmetric(vertical=10, horizontal=16),
              border_radius=4,
              cursor="pointer",
              font_size=16,
              margin=me.Margin.symmetric(vertical=8)
          )
      )
    
    if state.loading:
        me.progress_spinner(style=me.Style(margin=me.Margin.top(24)))
    
    if state.error_message:
      with me.box(style=me.Style(background="#f8d7da", border=me.Border.all(me.BorderSide(width=1, color="#f5c6cb", style="solid")), color="#721c24", padding=me.Padding.all(12), border_radius=4)):
        me.text(text=f"Error: {state.error_message}", type="body-1")
    
    if state.article_title:
      me.text(text=state.article_title, type="headline-3")
    
    if state.article_text:
      me.textarea(value=state.article_text, readonly=True, rows=20, style=me.Style(
          width="100%",
          font_size=16,
          padding=me.Padding.all(12),
          border=me.Border.all(me.BorderSide(width=1, color="#ccc", style="solid")),
          border_radius=4,
          font_family="inherit",
          white_space="pre-wrap"
      ))